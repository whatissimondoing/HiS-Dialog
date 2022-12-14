export CUDA_VISIBLE_DEVICES=0

VERSION=fused
BACKBONE=t5-base
TASK=e2e
DATASET=multiwoz
SEED=763904
PREFIX=50
MEMO="fused"

python main.py \
  -run_type predict \
  -backbone ${BACKBONE} \
  -dataset ${DATASET} \
  -seed ${SEED} \
  -task ${TASK} \
  -version ${VERSION} \
  -prefix_len ${PREFIX} \
  -ckpt outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}/ckpt-epoch9 \
  -output results.json \
  -batch_size 98
