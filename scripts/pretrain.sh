#!/bin/bash
gpu=0
ARGS=${@:2}

python train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu $gpu \
  $ARGS
