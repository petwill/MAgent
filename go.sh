#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=200

# reset
#rm -f mygather.log
#rm -f ./build/render/mygather/*

# original game
#python3 examples/train_mygather.py \
  #--map_size $MAP_SIZE \
  #--train \
  #--n_round 2999 \
  #--save_every 99 \
  #--render_every 9 \
  #--name mygather

# train diminishing game
python3 examples/train_mygather.py \
  --map_size $MAP_SIZE \
  --train \
  --n_round 2999 \
  --save_every 99 \
  --render_every 9 \
  --name mygather_diminishing \
  --diminishing

#python3 scripts/plot_log.py mygather.log 1 2

# original game
#python3 examples/train_mygather.py \
  #--map_size $MAP_SIZE \
  #--load_from 199 \
  #--n_round 10 \
  #--render_every 1 \
  #--name mygather 

# diminishing game
python3 examples/train_mygather.py \
  --map_size $MAP_SIZE \
  --load_from 197 \
  --n_round 10 \
  --render_every 1 \
  --name mygather_diminishing \
  --diminishing

