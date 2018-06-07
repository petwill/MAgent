"""
Train agents to gather food
"""

import argparse
import logging as log
import time
import random
from random import shuffle
import collections
import numpy as np
import os
import json

import magent
from magent.builtin.mx_model import DeepQNetwork as RLModel
# change this line to magent.builtin.tf_model to use tensorflow


def load_config(size, diminishing):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": size, "map_height": size})
    cfg.set({"embedding_size":1})
    cfg.set({"minimap_mode": True})

    agent = cfg.register_agent_type(
        name="agent_diminishing" if diminishing else "agent",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(20), 'attack_range': gw.CircleRange(1),
              'damage': 25, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    agent_strong = cfg.register_agent_type(
        name="agent_strong_diminishing" if diminishing else "agent_strong",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 6,
              'view_range': gw.CircleRange(20), 'attack_range': gw.CircleRange(1),
              'damage': 25, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    food = cfg.register_agent_type(
        name='food',
        attr={'width': 1, 'length': 1, 'hp': 20, 'speed': 0,
              'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
              'kill_reward': 50})

    g_f = cfg.add_group(food)
    g_s = cfg.add_group(agent)
    g_z = cfg.add_group(agent_strong)

    # a for agent
    a = gw.AgentSymbol(g_s, index='any')
    # b for food
    b = gw.AgentSymbol(g_f, index='any')
    # c for strong agent
    c = gw.AgentSymbol(g_z, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=2.5)
    # cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=a, value=5)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=2.5)
    
    return cfg


def generate_map(env, map_size, food_handle, player_handles):
    center_x, center_y = map_size // 2, map_size // 2

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])
    
    def add_random(pos, num):
        for _ in range(num):
            pos.append([random.randint(1, map_size-2), random.randint(1, map_size-2)])

    # pos = []
    # add_square(pos, map_size * 0.9, 3)
    # add_square(pos, map_size * 0.8, 4)
    # add_square(pos, map_size * 0.7, 6)
    # add_random(pos, 5)
    # shuffle(pos)

    def remove_duplicates(lst):
        seen = set()
        output = []
        for ob in lst:
            ob = tuple(ob)
            if ob in seen:
                continue
            output.append(list(ob))
            seen.add(ob)
        return output


    # sz = len(pos)//4
    # env.add_agents(player_handles[0], method="custom", pos=pos[:-sz])
    env.add_agents(player_handles[0], method="custom", pos=[[1,1], [1,map_size-2], [map_size-2, map_size-2], [map_size-2, 1]])
    # env.add_agents(player_handles[1], method="custom", pos=pos[-sz:])
    env.add_agents(player_handles[1], method="custom", pos=[[(map_size-1)/2, (map_size-1)/2]])

    player_pos = [[1,1], [1,map_size-2], [map_size-2, map_size-2], [map_size-2, 1], [(map_size-1)/2, (map_size-1)/2]]

    # food
    pos = []
    add_random(pos, 1)
    while pos[0] in player_pos:
        pos = []
        add_random(pos, 1)
    pos = remove_duplicates(pos)
    # mx = my = map_size/2
    # pos = [[mx,my], [mx-1, my], [mx+1, my], [mx, my-1], [mx, my+1]]
    # add_square(pos, map_size * 0.65, 10)
    # add_square(pos, map_size * 0.6,  1)
    # add_square(pos, map_size * 0.55, 10)
    # add_square(pos, map_size * 0.5,  1)
    # add_square(pos, map_size * 0.45, 3)
    # add_square(pos, map_size * 0.4, 1)
    # add_square(pos, map_size * 0.3, 1)
    # add_square(pos, map_size * 0.3 - 2, 1)
    # add_square(pos, map_size * 0.3 - 4, 1)
    # add_square(pos, map_size * 0.3 - 6, 1)
    print(pos)
    env.add_agents(food_handle, method="custom", pos=pos)


def play_a_round(env, map_size, food_handle, player_handles, models, train_id=-1,
                 print_every=10, record=False, render=False, eps=None, diminishing=False):
    env.reset()
    generate_map(env, map_size, food_handle, player_handles)

    step_ct = 0
    done = False

    pos_reward_ct = set()

    n = len(player_handles)

    history = collections.defaultdict(list)
    total_rewards = [0 for _ in range(n)]
    rewards = [None for _ in range(n)]
    alives = [None for _ in range(n)]
    prev_pos = [None for _ in range(n)]
    cur_pos  = [None for _ in range(n)]
    obs  = [None for _ in range(n)]
    ids  = [None for _ in range(n)]
    acts = [None for _ in range(n)]
    nums = [env.get_num(handle) for handle in player_handles]
    sample_buffer = [magent.utility.EpisodesBuffer(capacity=5000) for handle in player_handles]

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()

    #####
    # diminishing reward shaping config
    #####
    backpeak = 10
    thresh = 3
    ng = -100

    while not done or True:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(player_handles[i])
            ids[i] = env.get_agent_id(player_handles[i])
            prev_pos[i] = env.get_pos(player_handles[i])
            ##########
            # add custom feature
            ########
            for j in range(len(ids[i])):
                obs[i][1][j, 0] = sum(history[ids[i][j]][-backpeak:])

            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            env.set_action(player_handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n):
            rewards[i] = env.get_reward(player_handles[i])
            alives[i] = env.get_alive(player_handles[i])
            cur_pos[i] = env.get_pos(player_handles[i])
            total_rewards[i] += (np.array(rewards[i]) > 4).sum() 

        # respawn food
        # food_left = env.get_alive(food_handle)

        if done:
            print('here')
            player_pos = cur_pos[0].tolist() + cur_pos[1].tolist() + env.get_pos(food_handle).tolist()
            print(player_pos)
            
            pos = [random.randint(1, map_size-2), random.randint(1, map_size-2)]
            while pos in player_pos:
                pos = [random.randint(1, map_size-2), random.randint(1, map_size-2)]

            env.add_agents(food_handle, method="custom", pos=[pos])
            print(pos, env.get_pos(food_handle))

        # reward shaping for hunter prey scenario
        shaping=True
        if shaping and not done:
            food_pos = env.get_pos(food_handle)
            assert food_pos.shape[0] == 1
            food_pos = food_pos[0]
            print(rewards)
            # input()
            
            rewards[1][0] += 10*float(1/(abs(food_pos[0]-cur_pos[1][0][0])+abs(food_pos[1]-cur_pos[1][0][1])) - \
                                   1/(abs(food_pos[0]-prev_pos[1][0][0])+abs(food_pos[1]-prev_pos[1][0][1])))
            print(rewards)

        if diminishing:
            # implement reward shaping here
            thresh = 3
            for i in range(n):
                cnt = 0
                for idx, id in enumerate(ids[i]):
                    ori_reward = rewards[i][idx]
                    history[id].append(int(ori_reward > 4))
                    xx = sum(history[id][-backpeak:])
                    rewards[i][idx] = ori_reward if xx < thresh else ng
                    if xx >= thresh:
                        cnt += 1
                print("agent_strong" if i else "agent", cnt)

        collaboration = True
        if collaboration:
            for i in range(n):
                s = sum(rewards[i])
                for j in range(len(rewards[i])):
                    rewards[i][j] = s

                    # if ori_reward > 4:
                        # print("agent_strong" if i else "agent", ori_reward, xx, rewards[i][idx])
        # sample
        step_reward = 0
        if train_id != -1:
            for i in range(n):
                sample_buffer[i].record_step(ids[i], obs[i], acts[i], rewards[i], alives[i])
                # step_reward = sum(rewards)

        # render
        if render:
            env.render()

        """
        for id, r in zip(ids[0], rewards):
            if r > 0.05 and id not in pos_reward_ct:
                pos_reward_ct.add(id)
        """
        # clear dead agents
        env.clear_dead()

        """
        # stats info
        for i in range(n):
            nums[i] = env.get_num(player_handles[i])
        food_num = env.get_num(food_handle)

        if step_ct % print_every == 0:
            print("step %3d,  train %d,  num %s,  reward %.2f,  total_reward: %.2f, non_zero: %d" %
                  (step_ct, train_id, [food_num] + nums, step_reward, total_reward, len(pos_reward_ct)))
        """
        step_ct += 1

        if step_ct > 350:
            break

    """
    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    if record:
        with open("reward-hunger.txt", "a") as fout:
            fout.write(str(nums[0]) + "\n")

    """
    # train
    total_loss = value = 0
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        for i in range(n):
            total_loss, value = models[i].train(sample_buffer[i], print_every=250)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    """
    def to_json(data, filename):
        print(data, filename)
        with open(filename, 'w') as f:
            json.dump(data, f)
    """
    """
    if train_id == -1:
        if diminishing:
            dirname = "mygather_diminishing_history"
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            np.save('{}/{}.npy'.format(dirname, np.random.randint(10000)), history)
        else:
            dirname = "mygather_history"
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            np.save('{}/{}.npy'.format(dirname, np.random.randint(10000)), history)
    """
    
    return total_loss, total_rewards, value, len(pos_reward_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=1500)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--map_size", type=int, default=200)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="mygather")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--diminishing", action="store_true")
    args = parser.parse_args()

    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init env
    env = magent.GridWorld(load_config(size=args.map_size, diminishing=args.diminishing))
    render_dir = "build/render/"+args.name
    env.set_render_dir(render_dir)

    handles = env.get_handles()
    food_handle = handles[0]
    player_handles = handles[1:]

    # sample eval observation set
    eval_obs = [None for handle in player_handles]
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, food_handle, player_handles)
        eval_obs = magent.utility.sample_observation(env, player_handles, 0, 2048, 500)

    # load models
    models = [
        RLModel(env, player_handles[0], "agent_diminishing" if args.diminishing else "agent",
                batch_size=512, memory_size=2 ** 19, target_update=1000,
                train_freq=4, eval_obs=eval_obs[0]),
        RLModel(env, player_handles[1], "agent_strong_diminishing" if args.diminishing else "agent_strong",
                batch_size=512, memory_size=2 ** 19, target_update=1000,
                train_freq=4, eval_obs=eval_obs[1])
    ]

    # load saved model
    save_dir = args.name + "_save_model"
    
    if args.load_from is not None:
        start_from = args.load_from
        print("load models...")
        for model in models:
            model.load(save_dir, start_from)
    else:
        start_from = 0

    # print debug info

    print(args)
    for i in range(len(player_handles)):
        print('view_space', env.get_view_space(player_handles[i]))
        print('feature_space', env.get_feature_space(player_handles[i]))
        print('action_space', env.get_action_space(player_handles[i]))
        print('view2attack', env.get_view2attack(player_handles[i]))
    # input()

    if args.record:
        pass
        """
        for k in range(4, 999 + 5, 5):
            eps = 0
            for model in models:
                model.load(save_dir, start_from)
                play_a_round(env, args.map_size, food_handle, player_handles, models,
                             -1, record=True, render=False,
                             print_every=args.print_every, eps=eps)
        """
    else:
        # play
        start = time.time()
        train_id = 0 if args.train else -1
        for k in range(start_from, start_from + args.n_round):
            tic = time.time()
            eps = magent.utility.piecewise_decay(k, [0, 400, 1000], [1.0, 0.2, 0.05]) if not args.greedy else 0
            loss, reward, value, pos_reward_ct = \
                    play_a_round(env, args.map_size, food_handle, player_handles, models,
                                 train_id, record=False,
                                 render=args.render or (k+1) % args.render_every == 0,
                                 print_every=args.print_every, eps=eps, diminishing=args.diminishing)
            log.info("round %d\t loss: %.3f\t reward1: %.2f\t reward2: %.2f\t value: %.3f\t pos_reward_ct: %d"
                     % (k, loss, reward[0], reward[1], value, pos_reward_ct))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

            if (k + 1) % args.save_every == 0 and args.train:
                print("save models...")
                for model in models:
                    model.save(save_dir, k)
