#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=20
N_ROUND=15000

rm -f collab_raw.log
rm -f collab_origin.log

# original game
python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round $N_ROUND \
  --save_every 1000 \
  --name collab_raw --train

python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round 100 \
  --load_from 14999 \
  --name collab_raw --log

python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round $N_ROUND \
  --save_every 1000 \
  --name collab_origin --train --given

python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round 100 \
  --load_from 14999 \
  --name collab_origin --log --given
