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

total_num = 1000
total_food_num = 300

def load_config(size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": size, "map_height": size})
    # cfg.set({"embedding_size":23 + 33})
    cfg.set({"embedding_size": 2 + total_num*2 + total_food_num*2})
    cfg.set({"minimap_mode": True})

    agent = cfg.register_agent_type(
        name="agent",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(1),
              'damage': 12, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              })

    agent_strong = cfg.register_agent_type(
        name="agent_strong",
        attr={'width': 1, 'length': 1, 'hp': 300, 'speed': 3,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(1),
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
    g_z = cfg.add_group(agent_strong)

    # a for agent
    a = gw.AgentSymbol(g_s, index='any')
    # b for food
    b = gw.AgentSymbol(g_f, index='any')
    # c for strong agent
    c = gw.AgentSymbol(g_z, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=.5)
    cfg.add_reward_rule(gw.Event(c, 'attack', b), receiver=c, value=.5)

    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=-1)
    cfg.add_reward_rule(gw.Event(a, 'attack', c), receiver=a, value=-1)

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


    # player_pos = []
    # add_random(player_pos, 100)
    # player_pos = remove_duplicates(player_pos)
    # sz = len(player_pos)//4
    # env.add_agents(player_handles[0], method="custom", pos=player_pos[:-sz])
    # env.add_agents(player_handles[1], method="custom", pos=player_pos[-sz:])

    # player_pos = [[1,1], [1,map_size-2], [map_size-2, map_size-2], [map_size-2, 1], [(map_size-1)//2, (map_size-1)//2]]
    # env.add_agents(player_handles[0], method="custom", pos=[[1,1], [1,map_size-2], [map_size-2, map_size-2], [map_size-2, 1]])
    env.add_agents(player_handles[1], method="random", n=total_num-args.coop_num)
    env.add_agents(player_handles[0], method="random", n=args.coop_num)


    # food
    # pos = []
    # add_random(pos, 1)
    # while pos[0] in player_pos:
        # pos = []
        # add_random(pos, 1)
    # pos = remove_duplicates(pos)
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
    # print(pos)
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
    while not done:
        nums = [env.get_num(handle) for handle in player_handles]
        if nums != [args.coop_num, total_num-args.coop_num]:
            break


        # get observation
        for i in range(n):
            obs[i] = env.get_observation(player_handles[i])
            ids[i] = env.get_agent_id(player_handles[i])
            prev_pos[i] = env.get_pos(player_handles[i])

        for i in [1, 0]:
            ##########
            # add custom feature
            ########
            if args.diminishing:
                for j in range(len(ids[i])):
                    obs[i][1][j, 0] = sum(history[ids[i][j]][-backpeak:])

            # give ID embedding
            cnt = 5

            food_positions = env.get_pos(food_handle).tolist()
            assert len(food_positions) == total_food_num
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

            assert cnt == 5 + total_num * 2 + total_food_num * 2

            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            env.set_action(player_handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n):
            rewards[i] = env.get_reward(player_handles[i])
            alives[i] = env.get_alive(player_handles[i])
            cur_pos[i] = env.get_pos(player_handles[i])
            total_rewards[i] += (np.array(rewards[i]) > .8).sum()


        # if args.adversarial and args.load_from is None:
        if args.adversarial:
            for i in range(args.coop_num):
                rewards[0][i] -= args.coe * sum(rewards[1])/len(rewards[1])

        if args.diminishing:

            for i in range(n):
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

        """
        for id, r in zip(ids[0], rewards):
            if r > 0.05 and id not in pos_reward_ct:
                pos_reward_ct.add(id)
        """

        # clear dead agents
        env.clear_dead()

        # respawn
        food_num = env.get_num(food_handle)
        if total_food_num > food_num:
            env.add_agents(food_handle, method="random", n=(total_food_num-food_num))
        """
        # stats info
        for i in range(n):
            nums[i] = env.get_num(player_handles[i])
        food_num = env.get_num(food_handle)

        if step_ct % print_every == 0:
            print("step %3d,  num %s,  step_reward %s" %
                  (step_ct, [food_num] + nums, step_reward))
        """
        step_ct += 1

        if train_id != -1:
            if step_ct > 50:
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
    return total_loss, total_rewards, value, len(pos_reward_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2000000)
    parser.add_argument("--render_every", type=int, default=10000000)
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
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--given", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--coe", type=float)
    parser.add_argument("--coop_num", type=int, default=40)
    args = parser.parse_args()

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
        RLModel(env, player_handles[1], "agent_strong",
                batch_size=512, memory_size=2 ** 19, target_update=100,
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
            log.info("round %d\t loss: %.3f\t reward1: %.2f\t reward2: %.2f\t value: %.3f\t pos_reward_ct: %d"
                     % (k, loss, reward[0]/args.coop_num, reward[1]/(total_num-args.coop_num), value, pos_reward_ct))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))
            if args.log:
                f.write('{},{}\n'.format(reward[0], reward[1]))
                f.flush()

            if (k + 1) % args.save_every == 0 and args.train:
                print("save models...")
                for model in models:
                    model.save(save_dir, k)

        # print(np.mean(reward0), np.std(reward0))
        # print(np.mean(reward1), np.std(reward1))
        if args.log:
            f.close()

