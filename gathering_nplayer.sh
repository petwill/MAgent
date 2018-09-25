#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=21
N_ROUND=100
size0=$1
size1=$2

 #original game
python3 old_collab.py \
  --map_size $MAP_SIZE \
  --n_round $N_ROUND \
  --save_every 1000 \
  --render_every 100 \
  --load_from 13999 \
  --log \
  --name old_collab-$size0-$size1 --size0 $size0 --coe 1
