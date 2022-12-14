export CUDA_VISIBLE_DEVICES=0

VERSION=fused
BACKBONE=t5-base
TASK=e2e
SEED=763904  # 626521 for 2.0 and 763904 for fused
DATASET=multiwoz

if [ $DATASET = 'multiwoz' ]; then
  EPOCH=12 # 10 for 2.0 and 12 for FusedChat
  BATCH_SIZE=12
  LOGGIN_STEP=500
  PREFIX=50
  TEMPER=0.3
  ADD_CL=1
  ADD_PREFIX=1
  TRAIN_RATIO=1.0
  LR=6e-4  # 6e-4 for FusedChat, 5e-4 for 2.0
  KL_COEFF=1.0
  MEMO="test"
  MODEL_DIR=outputs/${DATASET}_${VERSION}_${BACKBONE}_${TASK}_${MEMO}_${CUDA_VISIBLE_DEVICES}
fi

python main.py \
  -backbone ${BACKBONE} \
  -dataset ${DATASET} \
  -task ${TASK} \
  -batch_size ${BATCH_SIZE} \
  -seed ${SEED} \
  -epochs $EPOCH \
  -learning_rate ${LR} \
  -version ${VERSION} \
  -run_type train \
  -add_cl_loss ${ADD_CL} \
  -add_adapter ${ADD_PREFIX} \
  -temper ${TEMPER} \
  -train_ratio ${TRAIN_RATIO} \
  -output results.json \
  -prefix_len ${PREFIX} \
  -kl_loss_coeff ${KL_COEFF} \
  -log_frequency ${LOGGIN_STEP} \
  -model_dir $MODEL_DIR/
