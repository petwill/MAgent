#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=200

# train config
python3 train_mygather.py \
  --map_size $MAP_SIZE \
  --n_round 2000 \
  --render_every 100 \
  --save_every 100 \
  --name mygather \
  --train
 
#python3 train_mygather.py \
  #--map_size $MAP_SIZE \
  #--n_round 2000 \
  #--render_every 100 \
  #--save_every 100 \
  #--diminishing \
  #--name mygather_diminishing \
  #--train


#python3 train_mygather.py \
  #--map_size $MAP_SIZE \
  #--load_from 899 \
  #--n_round 10 \
  #--render_every 1 \
  #--name mygather_diminishing  --diminishing

python3 train_mygather.py \
  --map_size $MAP_SIZE \
  --load_from 1099 \
  --n_round 10 \
  --render_every 1 \
  --name mygather
