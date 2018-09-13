#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=30
NAME=collab_diminishing0

# reset
#rm -f $NAME.log
#rm -f ./build/render/$NAME/*

# original game
python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round 1000 \
  --render_every 100 \
  --save_every 100 \
  --name $NAME \
  --train --diminishing
