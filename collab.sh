#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=20

NAME=$1
size0=$2
size1=$3

# reset
rm -f $NAME.log
rm -f ./build/render/$NAME/*

# original game
python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round 5000 \
  --render_every 100 \
  --save_every 100 \
  --name $NAME \
  --size0 $size0 --size1 $size1 --foodnum 50 \
  --train --adversarial
