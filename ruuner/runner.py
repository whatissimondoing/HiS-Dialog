import copy
import glob
import math
import os
import re
import shutil
import time
from abc import *
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput

from evaluator.evaluator import MultiWozEvaluator
from model.model import T5VAE
from dataset.reader import (MultiWOZIterator, MultiWOZReader, MultiWOZDataset, CollatorTrain)
from utils import definitions
from utils.io_utils import get_or_create_logger, save_json

from transformers.adapters.configuration import PrefixTuningConfig
import fitlog

logger = get_or_create_logger(__name__)


class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.span_loss = 0.0
        self.resp_loss = 0.0
        self.cl_loss = 0.0
        self.kl_loss = 0.0
        self.mode_loss = 0.0
        self.bow_loss = 0.0

        self.belief_correct = 0.0
        self.span_correct = 0.0
        self.resp_correct = 0.0
        self.mode_correct = 0.0

        self.belief_count = 0.0
        self.span_count = 0.0
        self.resp_count = 0.0
        self.mode_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        if "belief" in step_outputs:
            self.belief_loss += step_outputs["belief"]["loss"]
            self.belief_correct += step_outputs["belief"]["correct"]
            self.belief_count += step_outputs["belief"]["count"]
            do_belief_stats = True
        else:
            do_belief_stats = False

        if "span" in step_outputs:
            self.span_loss += step_outputs["span"]["loss"]
            self.span_correct += step_outputs["span"]["correct"]
            self.span_count += step_outputs["span"]["count"]
            do_span_stats = True
        else:
            do_span_stats = False

        if "mode" in step_outputs:
            self.mode_loss += step_outputs["mode"]["loss"]
            self.mode_correct += step_outputs["mode"]["correct"]
            self.mode_count += step_outputs["mode"]["count"]
            do_mode_stats = True
        else:
            do_mode_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.cl_loss += step_outputs["resp"]["cl_loss"]
            self.kl_loss += step_outputs["resp"]["kl_loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]
            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_belief_stats, do_span_stats, do_resp_stats, do_mode_stats)

    def info_stats(self, data_type, global_step, do_belief_stats=False, do_span_stats=False, do_resp_stats=False, do_mode_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        if do_belief_stats:
            try:
                belief_ppl = math.exp(self.belief_loss / self.belief_count)
            except:
                belief_ppl = 0
                print("belief loss and belief count is {}, {}".format(self.belief_loss, self.belief_count))
            belief_acc = (self.belief_correct / self.belief_count) * 100

            self.summary_writer.add_scalar(
                "{}/belief_loss".format(data_type), self.belief_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/belief_ppl".format(data_type), belief_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/belief_acc".format(data_type), belief_acc, global_step=global_step)

            if data_type == "train":
                fitlog.add_metric({"train": {"belief_acc": belief_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"belief_acc": belief_acc}}, step=global_step)

            belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(self.belief_loss, belief_ppl, belief_acc)
        else:
            belief_info = ""

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100
            if data_type == "train":
                fitlog.add_loss(self.resp_loss, name="loss", step=global_step)
                fitlog.add_loss(self.kl_loss, name="kl_loss", step=global_step)
                fitlog.add_loss(self.cl_loss, name="cl_loss", step=global_step)
                fitlog.add_metric({"train": {"resp_acc": resp_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"resp_acc": resp_acc}}, step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_kl_loss".format(data_type), self.kl_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_cl_loss".format(data_type), self.cl_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_bow_loss".format(data_type), self.bow_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; kl_loss {1:.3f}; cl_loss {2:.3f}; bow_loss {3:.3f} ppl {4:.2f}; acc {5:.2f}".format(
                self.resp_loss, self.kl_loss, self.cl_loss, self.bow_loss, resp_ppl, resp_acc)

        else:
            resp_info = ""

        if do_span_stats:
            if self.span_count == 0:
                span_acc = 0.0
            else:
                span_acc = (self.span_correct / self.span_count) * 100

            self.summary_writer.add_scalar(
                "{}/span_loss".format(data_type), self.span_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/span_acc".format(data_type), span_acc, global_step=global_step)

            span_info = "[span] loss {0:.2f}; acc {1:.2f};".format(
                self.span_loss, span_acc)

        else:
            span_info = ""

        if do_mode_stats:
            if self.mode_count == 0:
                mode_acc = 0.0
            else:
                mode_acc = (self.mode_correct / self.mode_count) * 100

            self.summary_writer.add_scalar(
                "{}/mode_loss".format(data_type), self.mode_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/mode_acc".format(data_type), mode_acc, global_step=global_step)

            mode_info = "[mode] loss {0:.2f}; acc {1:.2f};".format(
                self.mode_loss, mode_acc)

        else:
            mode_info = ""

        logger.info(" ".join([common_info, belief_info, resp_info, span_info, mode_info]))

        self.init_stats()


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader

        self.pbar = None
        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional_decoder = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional_decoder = False
        else:
            model_path = self.cfg.backbone
            initialize_additional_decoder = True

        logger.info("Load models from {}".format(model_path))

        self.cfg.vocab_size = self.reader.vocab_size

        model = T5VAE.from_pretrained(model_path, cfg=self.cfg)  # TODO

        print("parameters before adapters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # PrefixTuning ---
        if self.cfg.add_adapter:
            if self.cfg.run_type == "train":
                config = PrefixTuningConfig(flat=False, prefix_length=self.cfg.prefix_len, encoder_prefix=True, cross_prefix=True)
                model.add_adapter("style-prefix", config=config)

                print("Prefix len: {}, parameters after adapters: {}".format(self.cfg.prefix_len,
                                                                             sum(p.numel() for p in model.parameters() if p.requires_grad)))
            else:
                model.load_adapter(model_path + "-adapter")
                model.set_active_adapters("style-prefix")

            # print(model)

        # End PrefixTuning ----

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional_decoder:
            model.initialize_additional_decoder()

        model.to(self.cfg.device)
        if self.cfg.num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank,
                                                        find_unused_parameters=True)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)

        if self.cfg.num_gpus > 1:
            models = self.model.module
        else:
            models = self.model

        # model = self.model

        models.save_pretrained(save_path)
        if self.cfg.add_adapter:
            models.save_adapter(save_path + "-adapter", 'style-prefix')

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # keep checkpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch * self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            # num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        self.pbar = tqdm(total=num_train_steps, desc="training")

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        label_mask = label.eq(pad_id).bool()
        pred[label_mask] = -1

        num_count = label.view(-1).ne(pad_id).long().sum()
        num_correct = torch.eq(pred.view(-1), label.view(-1)).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_modes(self, pred, label):

        num_count = label.shape[0]
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, global_step):
        raise NotImplementedError


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg.backbone, cfg.version)

        super(MultiWOZRunner, self).__init__(cfg, reader)

        self.iterator = MultiWOZIterator(reader, cfg.device)

        self.kl_scheduler = None

    # @profile
    def step_fn(self, global_step, inputs, span_labels, belief_labels, resp_labels, mode_labels=None):

        attention_mask = torch.where(inputs.eq(self.reader.pad_token_id), 0, 1)

        if self.reader.version == "fused":
            belief_outputs = self.model(input_ids=inputs,
                                        attention_mask=attention_mask,
                                        span_labels=span_labels,
                                        lm_labels=belief_labels,
                                        return_dict=False,
                                        add_auxiliary_task=self.cfg.add_auxiliary_task,
                                        decoder_type="belief")
        else:
            belief_outputs = self.model(input_ids=inputs,
                                        attention_mask=attention_mask,
                                        span_labels=span_labels,
                                        lm_labels=belief_labels,
                                        return_dict=False,
                                        add_auxiliary_task=self.cfg.add_auxiliary_task,
                                        decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        mode_loss, mode_pred = 0, None
        span_loss, span_pred = 0, None
        resp_loss = 0
        kl_loss, cl_loss = 0, 0
        bow_loss = 0.0

        if self.cfg.add_mode_predict:
            mode_loss = belief_outputs[2]
            mode_pred = belief_outputs[3]
        else:
            span_loss = belief_outputs[2]
            span_pred = belief_outputs[3]

        if self.cfg.task == "e2e":
            last_hidden_state = belief_outputs[5]

            encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)
            if self.cfg.version == "fused" and self.cfg.add_cl_loss:
                resp_outputs = self.model(attention_mask=attention_mask,
                                          encoder_outputs=encoder_outputs,
                                          lm_labels=resp_labels,
                                          mode_labels=mode_labels,
                                          return_dict=False,
                                          decoder_type="resp"
                                          )
            else:
                resp_outputs = self.model(attention_mask=attention_mask,
                                          encoder_outputs=encoder_outputs,
                                          lm_labels=resp_labels,
                                          return_dict=False,
                                          decoder_type="resp"
                                          )

            resp_loss = resp_outputs[0]
            resp_pred = resp_outputs[1]
            kl_loss = resp_outputs[-1]
            cl_loss = resp_outputs[-2]

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        if self.cfg.add_auxiliary_task:
            num_span_correct, num_span_count = self.count_tokens(
                span_pred, span_labels, pad_id=0)

        if self.cfg.add_mode_predict:
            num_mode_correct, num_mode_count = self.count_modes(
                mode_pred, mode_labels)

        loss = belief_loss

        if self.cfg.add_auxiliary_task and self.cfg.aux_loss_coeff > 0:
            loss += (self.cfg.aux_loss_coeff * span_loss)

        if self.cfg.add_mode_predict:
            loss += mode_loss

        if self.cfg.add_cl_loss:
            loss += cl_loss

        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)
            loss += self.kl_scheduler[global_step - 1] * kl_loss
            # loss += kl_loss

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_auxiliary_task:
            step_outputs["span"] = {"loss": span_loss.item(),
                                    "correct": num_span_correct.item(),
                                    "count": num_span_count.item()}

        if self.cfg.add_mode_predict:
            step_outputs["mode"] = {"loss": mode_loss.item(),
                                    "correct": num_mode_correct.item(),
                                    "count": num_mode_count
                                    }

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "kl_loss": kl_loss,
                                    "cl_loss": cl_loss,
                                    "bow_loss": bow_loss,
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

        return loss, step_outputs

    # @profile
    def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()
        global_step = reporter.global_step if reporter else 0

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            if self.cfg.version == "fused" and self.cfg.add_cl_loss:
                inputs, span_labels, belief_labels, resp_labels, mode_label = batch
                mode_labels = mode_label.to(self.cfg.device)
            else:
                inputs, span_labels, belief_labels, resp_labels = batch
                mode_labels = None
            inputs = inputs.to(self.cfg.device)
            span_labels = span_labels.to(self.cfg.device)
            belief_labels = belief_labels.to(self.cfg.device)
            resp_labels = resp_labels.to(self.cfg.device)

            # _, belief_labels, _ = labels
            if self.cfg.version == "fused" and self.cfg.add_cl_loss:
                loss, step_outputs = self.step_fn(global_step, inputs, span_labels, belief_labels, resp_labels, mode_labels)
            else:
                loss, step_outputs = self.step_fn(global_step, inputs, span_labels, belief_labels, resp_labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            self.pbar.update(1)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]
                # lr = 0

                if reporter is not None and self.cfg.local_rank in [0, -1]:
                    reporter.step(start_time, lr, step_outputs)

    def train(self):
        num_workers = 0
        train_dataset = MultiWOZDataset(self.cfg, self.reader, "train", self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size,
                                        num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains,
                                        train_ratio=self.cfg.train_ratio)
        train_sampler = DistributedSampler(train_dataset) if self.cfg.num_gpus > 1 else RandomSampler(train_dataset)
        collator = CollatorTrain(self.reader.pad_token_id, self.reader.tokenizer, self.cfg)
        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.cfg.batch_size, collate_fn=collator,
                                           num_workers=4 * self.cfg.num_gpus, pin_memory=True)

        dev_dataset = MultiWOZDataset(self.cfg, self.reader, "dev", self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size,
                                      num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)
        dev_sampler = DistributedSampler(dev_dataset) if self.cfg.num_gpus > 1 else RandomSampler(dev_dataset)
        self.dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=3 * self.cfg.batch_size, collate_fn=collator,
                                         num_workers=num_workers)

        num_training_steps_per_epoch = len(self.train_dataloader) // self.cfg.grad_accum_steps

        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch, self.cfg.batch_size)

        if self.cfg.local_rank in [0, -1]:
            reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)
        else:
            reporter = None

        self.kl_scheduler = frange_cycle_linear(n_iter=self.pbar.total)

        # best_combine_score = 0

        for epoch in range(1, self.cfg.epochs + 1):

            self.train_epoch(self.train_dataloader, optimizer, scheduler, reporter)

            if self.cfg.local_rank in [0, -1]:
                logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))
                self.save_model(epoch)

                if not self.cfg.no_validation:
                    self.cfg.ckpt = os.path.join(self.cfg.model_dir, "ckpt-epoch" + str(epoch))
                    self.validation(reporter.global_step)
                    if epoch in [2] or epoch > self.cfg.epochs - 5:
                        results = self.predict(reporter.global_step)
                        print(results)
                        # if results["score"] > best_combine_score:
                        #     best_combine_score = results["score"]
                        #     fitlog.add_best_metric({"test": results})

        self.pbar.close()

    def validation(self, global_step):
        self.model.eval()

        reporter = Reporter(1000000, self.cfg.model_dir)

        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, total=len(self.dev_dataloader), desc="Validation"):
                start_time = time.time()

                if self.cfg.version == "fused" and self.cfg.add_cl_loss:
                    inputs, span_labels, belief_labels, resp_labels, mode_label = batch
                    mode_labels = mode_label.to(self.cfg.device)
                else:
                    inputs, span_labels, belief_labels, resp_labels = batch
                    mode_labels = None
                inputs = inputs.to(self.cfg.device)
                span_labels = span_labels.to(self.cfg.device)
                belief_labels = belief_labels.to(self.cfg.device)
                resp_labels = resp_labels.to(self.cfg.device)

                if self.cfg.version == "fused" and self.cfg.add_cl_loss:
                    loss, step_outputs = self.step_fn(global_step, inputs, span_labels, belief_labels, resp_labels, mode_labels)
                else:
                    loss, step_outputs = self.step_fn(global_step, inputs, span_labels, belief_labels, resp_labels)

                reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

            do_belief_stats = True if "belief" in step_outputs else False
            do_span_stats = True if "span" in step_outputs else False
            do_resp_stats = True if "resp" in step_outputs else False
            do_mode_stats = True if "mode" in step_outputs else False

            reporter.info_stats("dev", global_step, do_belief_stats, do_span_stats, do_resp_stats, do_mode_stats)

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            # update bspn using span output
            if span_outputs is not None and input_ids is not None:
                span_output = span_outputs[i]
                input_id = input_ids[i]

                # print(self.reader.tokenizer.decode(input_id))
                # print(self.reader.tokenizer.decode(bspn))

                eos_idx = input_id.index(self.reader.eos_token_id)
                input_id = input_id[:eos_idx]

                span_result = {}

                bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

                span_output = span_output[:eos_idx]

                b_slot = None
                for t, span_token_idx in enumerate(span_output):
                    turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
                    turn_domain = domain_history[i][turn_id]

                    if turn_domain not in definitions.INFORMABLE_SLOTS:
                        continue

                    span_token = self.reader.span_tokens[span_token_idx]

                    if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
                        b_slot = span_token
                        continue

                    if turn_domain not in span_result:
                        span_result[turn_domain] = defaultdict(list)

                    if b_slot != span_token:
                        span_result[turn_domain][span_token] = [input_id[t]]
                    else:
                        span_result[turn_domain][span_token].append(input_id[t])

                    b_slot = span_token

                for domain, sv_dict in span_result.items():
                    for s, v_list in sv_dict.items():
                        value = v_list[-1]
                        span_result[domain][s] = self.reader.tokenizer.decode(
                            value, clean_up_tokenization_spaces=False)

                span_dict = copy.deepcopy(span_result)

                ontology = self.reader.db.extractive_ontology

                flatten_span = []
                for domain, sv_dict in span_result.items():
                    flatten_span.append("[" + domain + "]")

                    for s, v in sv_dict.items():
                        if domain in ontology and s in ontology[domain]:
                            if v not in ontology[domain][s]:
                                del span_dict[domain][s]
                                continue

                        if s == "destination" or s == "departure":
                            _s = "destination" if s == "departure" else "departure"

                            if _s in sv_dict and v == sv_dict[_s]:
                                if s in span_dict[domain]:
                                    del span_dict[domain][s]
                                if _s in span_dict[domain]:
                                    del span_dict[domain][_s]
                                continue

                        if s in ["time", "leave", "arrive"]:
                            v = v.replace(".", ":")
                            if re.match("[0-9]+:[0-9]+", v) is None:
                                del span_dict[domain][s]
                                continue
                            else:
                                span_dict[domain][s] = v

                        flatten_span.append("[value_" + s + "]")
                        flatten_span.append(v)

                    if len(span_dict[domain]) == 0:
                        del span_dict[domain]
                        flatten_span.pop()

                # print(flatten_span)

                # input()

                decoded["span"] = flatten_span

                constraint_dict = self.reader.bspn_to_constraint_dict(
                    self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

                if self.cfg.overwrite_with_span:
                    _constraint_dict = OrderedDict()

                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain in constraint_dict or domain in span_dict:
                            _constraint_dict[domain] = OrderedDict()

                        for slot in slots:
                            if domain in constraint_dict:
                                cons_value = constraint_dict[domain].get(slot, None)
                            else:
                                cons_value = None

                            if domain in span_dict:
                                span_value = span_dict[domain].get(slot, None)
                            else:
                                span_value = None

                            if cons_value is None and span_value is None:
                                continue

                            # priority: span_value > cons_value
                            slot_value = span_value or cons_value

                            _constraint_dict[domain][slot] = slot_value
                else:
                    _constraint_dict = copy.deepcopy(constraint_dict)

                bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
                    _constraint_dict)

                bspn_gen_with_span = self.reader.encode_text(
                    bspn_gen_with_span,
                    bos_token=definitions.BOS_BELIEF_TOKEN,
                    eos_token=definitions.EOS_BELIEF_TOKEN)

                decoded["bspn_gen_with_span"] = bspn_gen_with_span

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                # logger.warn("bos/eos resp token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self, global_step):
        self.model.eval()

        if self.cfg.num_gpus > 1:
            self.model = self.model.module

        batch_size_for_predict = 96 if self.cfg.run_type == "train" else self.cfg.batch_size

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, batch_size=batch_size_for_predict,
            num_gpus=self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains, num_dialogs=self.cfg.num_train_dialogs)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)
            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    context, _ = self.iterator.flatten_dial_history(
                        dial_history[t], [], len(turn["user"]), self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                # batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(batch_encoder_input_ids.eq(self.reader.pad_token_id), 0, 1)

                # belief tracking
                with torch.no_grad():
                    encoder_outputs = self.model(input_ids=batch_encoder_input_ids,
                                                 attention_mask=attention_mask,
                                                 return_dict=False,
                                                 encoder_only=True,
                                                 add_auxiliary_task=self.cfg.add_auxiliary_task)

                    span_outputs, encoder_hidden_states = encoder_outputs

                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states

                    # wrap up encoder outputs
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state)

                    belief_outputs = self.model.generate(encoder_outputs=encoder_outputs,
                                                         attention_mask=attention_mask,
                                                         eos_token_id=self.reader.eos_token_id,
                                                         max_length=100,
                                                         do_sample=self.cfg.do_sample,
                                                         num_beams=self.cfg.beam_size,
                                                         early_stopping=early_stopping,
                                                         temperature=self.cfg.temperature,
                                                         top_k=self.cfg.top_k,
                                                         top_p=self.cfg.top_p,
                                                         decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                if self.cfg.add_auxiliary_task:
                    pred_spans = span_outputs[1].cpu().numpy().tolist()

                    input_ids = batch_encoder_input_ids.cpu().numpy().tolist()
                else:
                    pred_spans = None
                    input_ids = None

                decoded_belief_outputs = self.finalize_bspn(
                    belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            if self.cfg.add_auxiliary_task:
                                bspn_gen = turn["bspn_gen_with_span"]
                            else:
                                bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    # print(dbpn)

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = self.iterator.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                resp_outputs = self.model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=200,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            resp_outputs = self.model.generate(
                                encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=200,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        if self.cfg.add_auxiliary_task:
                            pv_bspn = turn["bspn_gen_with_span"]
                        else:
                            pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == "e2e":
            bleu, success, match, distinct1, distinct2, distinct3, bleu_odd = evaluator.e2e_eval(
                results, eval_dial_list=eval_dial_list, add_auxiliary_task=self.cfg.add_auxiliary_task)

            score = 0.5 * (success + match) + bleu

            logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (match, success, bleu, score))
            fitlog.add_metric({"test": {"inform": match, "success": success, "bleu": bleu, "score": score}}, step=global_step)

            if self.reader.version == "fused":
                logger.info('Metrics for ODD: Distinct 1: %.3f, Distinct 2: %.3f, Distinct 3: %.3f, BLEU: %.3f', distinct1, distinct2, distinct3,
                            bleu_odd)
                fitlog.add_metric({"test": {"dist-1": distinct1, "dist-2": distinct2, "dist-3": distinct3, "bleu_odd": bleu_odd}}, step=global_step)
            return {"inform": match, "success": success, "bleu": bleu, "score": score, "dist-1": distinct1, "dist-2": distinct2, "dist-3": distinct3,
                    "bleu_odd": bleu_odd}
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results, add_auxiliary_task=self.cfg.add_auxiliary_task)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
