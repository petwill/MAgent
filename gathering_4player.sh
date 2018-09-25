#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=20
N_ROUND=100
NAME=$1

 #original game
python3 gathering_4player.py \
  --map_size $MAP_SIZE \
  --n_round $N_ROUND \
  --save_every 1000 \
  --render_every 1 \
  --load_from 14999 \
  --log \
  --name $NAME --coe 1
