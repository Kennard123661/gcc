#!/bin/bash
gpu=$1
ARGS=${@:2}

python train_ogo.py \
  --exp Pretrain \
  --model-path saved/ogo \
  --tb-path tensorboard \
  --gpu $gpu \
  $ARGS
