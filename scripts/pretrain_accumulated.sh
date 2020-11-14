#!/bin/bash
gpu=$1
ARGS=${@:2}

python train_accumulated.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu $gpu \
  $ARGS
