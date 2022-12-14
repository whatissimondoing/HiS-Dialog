# HiS-Dialog

This is code for the paper "Mitigating Negative Style Transfer in Hybrid Dialogue System".

## Environment setting

Our python version is 3.7.11.

The package to reproduce the results can be installed by running the following command.

```
pip install -r requirements.txt
```

## Data Preprocessing

For the experiments, we use MultiWOZ2.0, MultiWOZ2.1 and FusedChat.

- (MultiWOZ2.0) annotated_user_da_with_span_full.json: A fully annotated version of the original MultiWOZ2.0 data released by developers of Convlab
  available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).
- (MultiWOZ2.1) data.json: The original MultiWOZ 2.1 data released by researchers in University of Cambrige
  available [here](https://github.com/budzianowski/multiwoz/tree/master/data).
- (FusedChat) data.json: The original data can be downloaded [here](https://github.com/tomyoung903/FusedChat). We processed it to be consistent to
  MultiWOZ format.

The range for $VERSION is [ 2.0, 2.1, fused ]

> We have preprocessed the FusedChat and MultiWOZ2.0 and uploaded the data for fast reproduce, so this command can be skipped. If you want to preprocess the data by yourself, you can run the following command:

[//]: # (We use the preprocessing scripts implemented by [Zhang et al., 2020]&#40;https://arxiv.org/abs/1911.10484&#41;. Please refer to [here]&#40;https://github.com/thu-spmi/damd-multiwoz/blob/master/data/multi-woz/README.md&#41; for the details.)

```
python preprocess.py -version $VERSION
```

## Training

We uploaded the pre-processed FusedChat and MultiWOZ2.0 for fast reproduce.

- HiS-Dialog for FusedChat. We have set the hyper-parameters in the shell script to reproduce the reported results perfectly.

- We use ```fitlog``` as training monitor for visualization and results recoder, so please run the following command before training:
```shell
fitlog init
```

- For fast reproduce of FusedChat, you may check the following hyper-parameters in ```train.sh```:
```shell
VERSION=fused
SEED=763904
EPOCH=12
BATCH_SIZE=12
PREFIX=50
TEMPER=0.3
ADD_CL=1
ADD_PREFIX=1
LR=6e-4
```

- For fast reproduce of MultiWOZ2.0, you may check the following hyper-parameters in ```train.sh```:
```shell
VERSION=2.0
SEED=626521
EPOCH=10
BATCH_SIZE=12
PREFIX=50
TEMPER=0.3
ADD_CL=1
ADD_PREFIX=1
LR=5e-4

```

Checkpoints are saved after each epoch and only the latest five checkpoints are retained.

> Model runs on one GeForce 3090 GPU (24G) by default, and we didn't try it on CPU.

## Predict

The script ```predict.sh``` will make predictions and directly output the evaluation results based on the predictions.

- For prediction of FusedChat, you may to change the following hyper-parameters in ```predict.sh```:
```shell
VERSION=fused
SEED=763904
-ckpt outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}/ckpt-epoch9
```

If all goes well, the following results will be obtained.

|            | Inform | Success | BLEU  | Combined | DIST-1 | DIST-2 | DIST-3 | BLEU |
|------------|--------|---------|-------|----------|--------|--------|--------|------|
| HiS-Dialog | 92.60  | 84.30   | 17.76 | 106.21   | 0.03   | 0.12   | 0.23   | 9.25 |

- For prediction of MultiWOZ2.0, you may to change the following hyper-parameters in ```predict.sh```:
```shell
VERSION=2.0
SEED=626521
-ckpt outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}/ckpt-epoch6
```

If all goes well, the following results will be obtained.

|            | Inform | Success | BLEU  | Combined |
|------------|--------|---------|-------|----------|
| HiS-Dialog | 94.60  | 85.40   | 18.87 | 108.87   |


## Acknowledgements

This code is referenced from the [MTTOD](https://github.com/bepoetree/MTTOD) implementation, and we appreciate their contribution.

