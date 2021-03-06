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
#from magent.builtin.mx_model import AdvantageActorCritic as RLModel
# change this line to magent.builtin.tf_model to use tensorflow

def load_config(size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": size, "map_height": size})
    #cfg.set({"embedding_size":22 + 33})
    cfg.set({"embedding_size": 3 + 1 + total_num*2 + total_food_num*2})
    cfg.set({"minimap_mode": True})

    agent = cfg.register_agent_type(
        name="agent",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(2), 'attack_range': gw.CircleRange(1),
              'damage': 12, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    agent1 = cfg.register_agent_type(
        name="agent1",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(2), 'attack_range': gw.CircleRange(1),
              'damage': 12, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    agent_strong = cfg.register_agent_type(
        name="agent_strong",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(2), 'attack_range': gw.CircleRange(1),
              'damage': 25, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    food = cfg.register_agent_type(
        name='food',
        attr={'width': 1, 'length': 1, 'hp': 20, 'speed': 0,
              'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
              'kill_reward': 1})

    g_f = cfg.add_group(food)
    g_s = cfg.add_group(agent)
    g_s1 = cfg.add_group(agent1)
    g_z = cfg.add_group(agent_strong)

    # a for agent
    a = gw.AgentSymbol(g_s, index='any')
    # b for food
    b = gw.AgentSymbol(g_f, index='any')
    # c for strong agent
    c = gw.AgentSymbol(g_z, index='any')
    # 
    d = gw.AgentSymbol(g_s1, index='any')

    cfg.add_reward_rule(gw.Event(d, 'attack', b), receiver=d, value=.5)
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=.5)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=.5)

    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=-1)
    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=a, value=-1)
    cfg.add_reward_rule(gw.Event(a, 'attack', d), receiver=a, value=-1)
    cfg.add_reward_rule(gw.Event(d, 'attack', a), receiver=d, value=-1)
    cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=c, value=-1)
    cfg.add_reward_rule(gw.Event(d, 'attack', c), receiver=d, value=-1)

    return cfg


def generate_map(env, map_size, food_handle, player_handles):

    env.add_agents(player_handles[0], method="random", n=8)
    env.add_agents(player_handles[1], method="random", n=1)
    env.add_agents(player_handles[2], method="random", n=1)
    env.add_agents(food_handle, method="random", n=total_food_num)


def play_a_round(env, map_size, food_handle, player_handles, models, train_id=-1,
                 print_every=10, record=False, render=False, eps=None, args=None):

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
    sample_buffer = [magent.utility.EpisodesBuffer(capacity=5000) for handle in player_handles]

    print("===== sample =====")
    nums = [env.get_num(handle) for handle in player_handles]
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()

    #####
    # diminishing reward shaping config
    #####
    backpeak = 3
    thresh = 2
    ng = -100

    X_train = []
    y_train = []
    while True:
        nums = [env.get_num(handle) for handle in player_handles]
        if nums != [8, 1, 1]:
            break


        # get observation
        for i in range(n):
            obs[i] = env.get_observation(player_handles[i])
            ids[i] = env.get_agent_id(player_handles[i])
            prev_pos[i] = env.get_pos(player_handles[i])

        food_positions = env.get_pos(food_handle).tolist()
        assert len(food_positions) == total_food_num
        for i in range(n):
            ##########
            # add custom feature
            ########

            # give ID embedding
            cnt = 3

            if args.diminishing:
                for j in range(len(ids[i])):
                    obs[i][1][j, cnt] = sum(history[ids[i][j]][-backpeak:])

            cnt += 1

            for food_pos in food_positions:
                for j in range(len(ids[i])):
                    obs[i][1][j, cnt] = food_pos[0]
                    obs[i][1][j, cnt+1] = food_pos[1]

                cnt += 2

            # add feature, add coordinate between agents
            for k in range(n):
                for l in range(len(ids[k])):
                    obs[i][1][:, cnt] = prev_pos[k][l][0]
                    obs[i][1][:, cnt+1] = prev_pos[k][l][1]
                    cnt += 2

            assert cnt == 3 + 1 + total_food_num * 2 + total_num * 2

            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            # if i == 1:
                # X_train.append(obs[1][1][0])
                # y_train.append(acts[1][0])
            # print(acts[i])
            # input()
            env.set_action(player_handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n):
            rewards[i] = env.get_reward(player_handles[i])
            alives[i] = env.get_alive(player_handles[i])
            cur_pos[i] = env.get_pos(player_handles[i])
            total_rewards[i] += (np.array(rewards[i]) > .8).sum()


        # if args.adversarial and args.load_from is None:
        coop_idxs = [0,2]
        if args.adversarial:
            for i in coop_idxs:
                for j in range(len(ids[i])):
                    for k in range(3):
                        if k not in coop_idxs:
                            rewards[i][j] -= args.coe * sum(rewards[k])/(total_num-cnum)

        if args.diminishing:

            for i in coop_idxs:
                cnt = 0
                for idx, id in enumerate(ids[i]):
                    ori_reward = rewards[i][idx]
                    history[id].append(int(ori_reward > .8))
                    xx = sum(history[id][-backpeak:])
                    rewards[i][idx] = ori_reward if xx < thresh else ng
                    if xx >= thresh:
                        cnt += 1
                # print("agent_strong" if i else "agent", cnt)

        """
        if share_reward:
            for i in range(n):
                s = sum(rewards[i])
                for j in range(len(rewards[i])):
                    rewards[i][j] = s
        """
        # sample
        step_reward = [None for _ in range(n)]
        if train_id != -1:
            for i in range(n):
                sample_buffer[i].record_step(ids[i], obs[i], acts[i], rewards[i], alives[i])
                step_reward[i] = rewards[i]

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # respawn
        food_num = env.get_num(food_handle)
        env.add_agents(food_handle, method="random", n=total_food_num-food_num)
        step_ct += 1

        if train_id != -1:
            if step_ct > 1000:
                break
        else:
            if step_ct > 1000:
                break

    # train
    total_loss = value = 0
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        for i in range(n):
            total_loss, value = models[i].train(sample_buffer[i], print_every=250)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)


    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # print(X_train.shape, y_train.shape)
    # np.save('./tmp/{}_train.npz'.format(np.random.randint(10000)), X_train)
    # np.save('./tmp/{}_test.npz'.format(np.random.randint(10000)), y_train)
    # sum(total_loss) ?
    # return sum(total_loss), total_rewards, value, len(pos_reward_ct)
    print(total_rewards)
    return total_loss, total_rewards, value, len(pos_reward_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2000000)
    parser.add_argument("--render_every", type=int, default=10000000)
    parser.add_argument("--n_round", type=int, default=35000)
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
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--given", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--food_num", type=int, default=5)
    parser.add_argument("--total_num", type=int, default=10)
    parser.add_argument("--cnum", type=int, default=4)
    parser.add_argument("--coe", type=float)

    args = parser.parse_args()

    global total_food_num, total_num, cnum
    total_food_num = args.food_num 
    total_num = args.total_num
    cnum = args.cnum


    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init env
    env = magent.GridWorld(load_config(size=args.map_size))
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
        RLModel(env, player_handles[0], "agent",
                batch_size=512, memory_size=2 ** 19, target_update=100,
                train_freq=4, eval_obs=eval_obs[0]),

        RLModel(env, player_handles[1], "agent1",
                batch_size=512, memory_size=2 ** 19, target_update=100,
                train_freq=4, eval_obs=eval_obs[1]),

        RLModel(env, player_handles[2], "agent_strong",
                batch_size=512, memory_size=2 ** 19, target_update=100,
                train_freq=4, eval_obs=eval_obs[2])
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

    else:
        fname = args.name
        print(fname)
        if args.log:
            f = open(fname, 'a+')
            f.write('agent,agent_strong\n')

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
                                 print_every=args.print_every, eps=eps, args=args)
            log.info("round %d\t loss: %.3f\t reward1: %.2f\t reward2: %.2f\t reward3: %.2f\t value: %.3f\t pos_reward_ct: %d"
                     % (k, loss, reward[0]/8, reward[1], reward[2], value, pos_reward_ct))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))
            if args.log:
                f.write('{},{}\n'.format(reward[1], reward[2]))
                f.flush()

            if (k + 1) % args.save_every == 0 and args.train:
                print("save models...")
                for model in models:
                    model.save(save_dir, k)

        # print(np.mean(reward0), np.std(reward0))
        # print(np.mean(reward1), np.std(reward1))
        if args.log:
            f.close()

