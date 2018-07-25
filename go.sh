#!/bin/bash -ex
. ~/ENV/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

MAP_SIZE=20
#NAME=collab_shping

# reset
#rm -f $NAME.log
#rm -f ./build/render/$NAME/*

# original game
python3 train_collab.py \
  --map_size $MAP_SIZE \
  --n_round 5000 \
  --save_every 100 \
  --render_every 100 \
  --load_from 4999 \
  --name collab_shaping

# diminishing game
#python3 train_collab.py \
  #--load_from 4999 \
  #--map_size $MAP_SIZE \
  #--n_round 5000 \
  #--save_every 100 \
  #--render_every 100 \
  #--name collab_diminishing \
  #--diminishing

# train diminishing game
#python3 examples/train_mygather.py \
  #--map_size $MAP_SIZE \
  #--train \
  #--n_round 2000 \
  #--save_every 100 \
  #--render_every  10 \
  #--name mygather_50
  #--diminishing

# diminishing game
#python3 examples/train_mygather.py \
  #--map_size $MAP_SIZE \
  #--load_from 197 \
  #--n_round 10 \
  #--render_every 1 \
  #--name mygather_diminishing \
  #--diminishing

