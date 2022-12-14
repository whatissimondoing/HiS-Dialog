import copy

import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers import T5ForConditionalGeneration
from torch.nn import functional as F
from transformers.modeling_utils import Conv1D

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances.base_distance import BaseDistance


class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = F.gelu

    def forward(self, inputs, attention_mask=None):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations


class CosineDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, normalize_embeddings=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return 1 - torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return 1 - torch.sum(query_emb * ref_emb, dim=1)


class T5VAE(T5ForConditionalGeneration):
    def __init__(self, config, cfg):
        super(T5VAE, self).__init__(config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        self.low_size = config.d_model // 6

        self.averageSelfAttention_prior = AverageSelfAttention(config.d_model)
        self.mean_prior = Conv1D(config.d_model, config.d_model)
        self.logvar_prior = Conv1D(config.d_model, config.d_model)
        self.averageSelfAttention_post = AverageSelfAttention(config.d_model)
        self.mean_post = Conv1D(config.d_model, config.d_model)
        self.logvar_post = Conv1D(config.d_model, config.d_model)

        self.input_proj1 = nn.Linear(config.d_model, config.d_model, bias=False)
        self.input_proj2 = nn.Linear(config.d_model, config.d_model, bias=False)

        self.gate = nn.Linear(config.d_model, 2)
        self.gate_control = nn.Softmax(dim=-1)

        self.cfg = cfg
        if self.cfg.add_cl_loss:
            self.cl_loss_fct = losses.TripletMarginLoss(margin=self.cfg.temper,
                                                        swap=False,
                                                        smooth_loss=False,
                                                        triplets_per_anchor="all",
                                                        distance=CosineDistance(),
                                                        )
            # self.cl_loss_fct = losses.NTXentLoss(temperature=self.cfg.temper,
            #                                      distance=CosineDistance()
            #                                      )

        self.add_mode_predict = self.cfg.add_mode_predict
        if self.add_mode_predict:
            self.mode_classifier = nn.Linear(config.d_model, 2, bias=True)

    def initialize_additional_decoder(self):
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

    def initialize_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def predict_mode(self, encoder_hidden_states, attention_mask, mode_labels=None):
        # mode_loss, pred_mode, mode_logits = 0, None, None
        mask_for_fill = attention_mask.unsqueeze(-1).expand(-1, -1, encoder_hidden_states.shape[-1]).bool()
        masked_hidden_states = encoder_hidden_states.masked_fill(~mask_for_fill, -1e8)  # mask position of padding with -1e8
        pooled_hidden_state, _ = torch.max(masked_hidden_states, dim=1)
        mode_logits = self.mode_classifier(pooled_hidden_state)
        mode_loss_fct = nn.CrossEntropyLoss()
        mode_loss = mode_loss_fct(mode_logits.view(-1, mode_logits.size(-1)), mode_labels)
        pred_mode = torch.argmax(mode_logits, dim=-1)
        return mode_loss, pred_mode, mode_logits

    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "decoder_type": kwargs.get("decoder_type")}

    def prior(self, hidden_states, attention_mask):
        representations = self.averageSelfAttention_prior(hidden_states, attention_mask)
        prior_mean = self.mean_prior(representations)
        prior_logvar = self.logvar_prior(representations)
        return prior_mean, prior_logvar

    def posterior(self, hidden_states, attention_mask):
        representations = self.averageSelfAttention_post(hidden_states, attention_mask)
        post_mean = self.mean_post(representations)
        post_logvar = self.logvar_post(representations)
        return post_mean, post_logvar

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                span_labels=None,
                lm_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,
                mode_labels=None,
                add_auxiliary_task=False,
                decoder_type=None,
                step_ratio=0.0):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        mode_loss, pred_mode, mode_logits = 0, None, None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=return_dict)

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]

        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        if encoder_only:
            return (mode_loss, pred_mode, mode_logits), encoder_outputs

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        # -- newly added --
        cl_loss = 0

        prior_mean, prior_logvar = self.prior(encoder_hidden_states, attention_mask)
        posterior_mean, posterior_logvar = self.posterior(encoder_hidden_states, attention_mask)

        z1 = self.reparameterize(posterior_mean, posterior_logvar)
        z2 = self.reparameterize(posterior_mean, posterior_logvar)

        if decoder_type == "resp":
            if self.cfg.add_cl_loss:
                if mode_labels is not None:
                    # supervised cl
                    cl_loss += self.cl_loss_fct(torch.cat([z1, z2], dim=0), mode_labels.repeat(2))
                else:
                    # unsupervised cl
                    batch_size = z1.shape[0]
                    mode_labels = torch.arange(0, batch_size).repeat(2)
                    cl_loss += self.cl_loss_fct(torch.cat([z1, z2], dim=0), mode_labels)

        input_proj1 = self.input_proj1(z1)
        input_proj2 = self.input_proj2(z2)

        mask_for_fill = attention_mask.unsqueeze(-1).expand(-1, -1, encoder_hidden_states.shape[-1]).bool()
        masked_hidden_states = encoder_hidden_states.masked_fill(~mask_for_fill, -1e8)  # mask position of padding with -1e8
        pooled_hidden_state, _ = torch.max(masked_hidden_states, dim=1)  # max pooling (bsz, hs)
        z_weight = self.gate_control(self.gate(pooled_hidden_state))
        input_proj = z_weight[:, 0].unsqueeze(-1) * input_proj1 + z_weight[:, 1].unsqueeze(-1) * input_proj2
        encoder_hidden_states = encoder_hidden_states + input_proj.unsqueeze(1)

        hs = encoder_hidden_states * (self.model_dim ** -0.5)

        if self.add_mode_predict and mode_labels is not None:
            mode_loss, pred_mode, mode_logits = self.predict_mode(hs, attention_mask, mode_labels)

        # if self.cfg.add_cl_loss and mode_labels is not None:
        #     cl_loss += self.mode_contrastive(torch.cat([z1, z2], dim=0), mode_labels.repeat(2))

        if lm_labels is not None:
            kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar)
        else:
            kl_loss = 0
        # -- end newly added --

        decoder = self.decoder
        lm_head = self.lm_head

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = decoder(input_ids=decoder_input_ids,
                                  inputs_embeds=decoder_inputs_embeds,
                                  past_key_values=past_key_values,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  return_dict=return_dict,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states)

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (lm_loss, pred_lm) + (mode_loss, pred_mode, mode_logits, encoder_hidden_states) + (decoder_outputs[1:], cl_loss, kl_loss)

        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs


class T5Base(T5ForConditionalGeneration):
    def __init__(self, config, cfg):
        super(T5Base, self).__init__(config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        # 暂时注释
        # self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        # self.resp_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def initialize_additional_decoder(self):
        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False

        # 暂时注释
        # self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
        # self.resp_lm_head = type(self.lm_head)(self.config.d_model, self.config.vocab_size, bias=False)
        #
        # self.resp_decoder.load_state_dict(self.decoder.state_dict())
        # self.resp_lm_head.load_state_dict(self.lm_head.state_dict())

    def initialize_weights(self, modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def predict_span(self, encoder_hidden_states, attention_mask, span_labels=None):
        span_loss, pred_spans, span_logits = 0, None, None

        return span_loss, pred_spans, span_logits

    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,
                "decoder_type": kwargs.get("decoder_type")}

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                span_labels=None,
                lm_labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,
                add_auxiliary_task=None,
                decoder_type=None):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        span_loss, pred_spans, span_logits = 0, None, None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=return_dict)

            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]

            hs = encoder_hidden_states * (self.model_dim ** -0.5)

            if add_auxiliary_task:
                span_loss, pred_spans, span_logits = self.predict_span(
                    hs, attention_mask, span_labels)

        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state

        if encoder_only:
            return (span_loss, pred_spans, span_logits), encoder_outputs

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(lm_labels)

        # if decoder_type == "resp":
        #     decoder = self.resp_decoder
        #     lm_head = self.resp_lm_head
        #
        # else:
        #     decoder = self.decoder
        #     lm_head = self.lm_head

        # 尝试一下如果只用一个decoder的效果如何
        decoder = self.decoder
        lm_head = self.lm_head

        if past_key_values is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        decoder_outputs = decoder(input_ids=decoder_input_ids,
                                  inputs_embeds=decoder_inputs_embeds,
                                  past_key_values=past_key_values,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask,
                                  use_cache=use_cache,
                                  return_dict=return_dict,
                                  output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states)

        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = lm_head(sequence_output)

        lm_loss = None
        if lm_labels is not None:
            lm_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (lm_loss, pred_lm,) + (span_loss, pred_spans, span_logits, encoder_hidden_states) + decoder_outputs[1:]

        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=lm_loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.models.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
