import os
import csv
import json
import random
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from custom_reward import *
from agent import Agent
from train import TrainAgent
from matplotlib import pyplot as plt
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from simple_opponents.random_opponent import RandomOpponent, WeightedRandomOpponent
from ppo.ppo import PPO
from ppo.nnpytorch import FFN


ENV_CASE = {
    '5': 'rte_case5_example',
    'sand': 'l2rpn_case14_sandbox',
    'wcci': 'l2rpn_wcci_2020'
}

DATA_SPLIT = {
    '5': ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    'sand': (list(range(0, 40*26, 40)), list(range(1, 100*10+1, 100)), list(range(2, 100*10+2, 100))),
    'wcci': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
            list(range(2880, 2890)), [18, 241, 495, 738, 991, 1453, 1718, 1943, 2205, 2404])
}

MAX_FFW = {
    '5': 5,
    'sand': 26,
    'wcci': 26
}


def cli():
    parser = ArgumentParser()
    parser.add_argument('-c', '--case', type=str, default='wcci', choices=['sand', 'wcci', '5'])
    parser.add_argument('-gpu', '--gpuid', type=int, default=0)

    parser.add_argument('-hn', '--head_number', type=int, default=8,
                        help='the number of head for attention')
    parser.add_argument('-sd', '--state_dim', type=int, default=128,
                        help='dimension of hidden state for GNN')
    parser.add_argument('-nh', '--n_history', type=int, default=6,
                        help='length of frame stack')
    parser.add_argument('-do', '--dropout', type=float, default=0.)
    
    parser.add_argument('-r', '--rule', type=str, default='c', choices=['c', 'd', 'o', 'f'],
                        help='low-level rule (capa, desc, opti, fixed)')
    parser.add_argument('-thr', '--threshold', type=float, default=0.1,
                        help='[-1, thr) -> bus 1 / [thr, 1] -> bus 2')
    parser.add_argument('-dg', '--danger', type=float, default=0.9,
                        help='the powerline with rho over danger is regarded as hazardous')
    parser.add_argument('-m', '--mask', type=int, default=5,
                        help='this agent manages the substations containing topology elements over "mask"')
    parser.add_argument('-mhi', '--mask_hi', type=int, default=19)
    parser.add_argument('-mll', '--max_low_len', type=int, default=19)
    
    parser.add_argument('-l', '--last', action='store_true')
    parser.add_argument('-n', '--name', type=str, required=True)

    args = parser.parse_args()
    return args

def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[case]):
            with open(os.path.join(path, '%d_%d.json' % (i,j)), 'r', encoding='utf-8') as f:
                a = json.load(f)
                res[(i,j)] = (a['dn_played'], a['donothing_reward'], a['donothing_nodisc_reward'])
            if i >= 2880: break
    return res

def read_loss_json(path, chronics):
    losses = {}
    loads = {}
    chronics = list(set(chronics))
    for i in chronics:
        json_path = os.path.join(path, '%s.json' % i)
        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        losses[i] = res['losses']
        loads[i] = res['sum_loads']
    return losses, loads


if __name__ == '__main__':
    args = cli()

    # settings
    model_name = args.name
    print('model_name: ', model_name)

    OUTPUT_DIR = './result'
    DATA_DIR = './data'
    output_result_dir = os.path.join(OUTPUT_DIR, model_name)
    model_path = os.path.join(output_result_dir, 'model')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_name = ENV_CASE[args.case]
    env_path = os.path.join(DATA_DIR, env_name)
    chronics_path = os.path.join(env_path, 'chronics')
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.case]
    dn_json_path = os.path.join(env_path, 'json')

    # select chronics
    dn_ffw = read_ffw_json(dn_json_path, test_chronics, args.case)

    # when small number of train, test chronic
    ep_infos = {}
    if os.path.exists(dn_json_path):
        for i in list(set(test_chronics)):
            with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
                ep_infos[i] = json.load(f)

    mode = 'last' if args.last else 'best'

    env = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
                other_rewards={'loss': LossReward})
    env.deactivate_forecast()
    env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    env.parameters.NB_TIMESTEP_RECONNECTION = 12
    env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    print(env.parameters.__dict__)

    my_agent = Agent(env, **vars(args))
    mean = torch.load(os.path.join(env_path, 'mean.pt'))
    std = torch.load(os.path.join(env_path, 'std.pt'))
    my_agent.load_mean_std(mean, std)
    my_agent.load_model(model_path, mode)

    LINES = ['0_4_2', '10_11_11', '11_12_13', '12_13_14', '12_16_20', 
                    '13_14_15', '13_15_16', '14_16_17', '14_35_53', '15_16_21', 
                    '16_17_22', '16_18_23', '16_21_27', '16_21_28', '16_33_48', 
                    '16_33_49', '16_35_54', '17_24_33', '18_19_24', '18_25_35', 
                    '19_20_25', '1_10_12', '1_3_3', '1_4_4', '20_21_26', 
                    '21_22_29', '21_23_30', '21_26_36', '22_23_31', '22_26_39', 
                    '23_24_32', '23_25_34', '23_26_37', '23_26_38', '26_27_40', 
                    '26_28_41', '26_30_56', '27_28_42', '27_29_43', '28_29_44', 
                    '28_31_57', '29_33_50', '29_34_51', '2_3_0', '2_4_1', 
                    '30_31_45', '31_32_47', '32_33_58', '33_34_52', '4_5_55', 
                    '4_6_5', '4_7_6', '5_32_46', '6_7_7', '7_8_8', 
                    '7_9_9', '8_9_10', '9_16_18', '9_16_19']
    attack_period = 50
    attack_duration = 3
    hyperparameters = {
                    'lines_attacked': LINES,
                    'attack_duration': 3,
                    'attack_period': 50,
                    'danger': 0.9,
                    'state_dim': 1062
                  }

    opponent_ppo = PPO(env=env, agent=my_agent, policy_class=FFN, state_mean=mean, state_std=std, **hyperparameters)
    opponent_ppo.actor.load_state_dict(torch.load('./ppo_actor_kaist.pth'))
    opponent_ran = RandomOpponent(env.observation_space, env.action_space,
                          lines_to_attack=LINES, attack_period=attack_period,
                          attack_duration=attack_duration)
    opponent_wro = WeightedRandomOpponent(env.observation_space, env.action_space,
                          lines_to_attack=LINES, attack_period=attack_period,
                          attack_duration=attack_duration)
    #opponent = PPO(env=env, agent=agent, policy_class=FFN, state_mean=state_mean, state_std=state_std, **hyperparameters)
    #opponent.actor.load_state_dict(torch.load('./ppo_actor_kaist.pth'))
    trainer_def = TrainAgent(my_agent, None, env, env, device, dn_json_path, dn_ffw, ep_infos)
    trainer_ppo = TrainAgent(my_agent, opponent_ppo, env, env, device, dn_json_path, dn_ffw, ep_infos)
    trainer_ran = TrainAgent(my_agent, opponent_ran, env, env, device, dn_json_path, dn_ffw, ep_infos)
    trainer_wro = TrainAgent(my_agent, opponent_wro, env, env, device, dn_json_path, dn_ffw, ep_infos)

    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    print("-" * 20 + " No Opponent " + "-" * 20)
    trainer_def.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/no_opp_", mode)
    print("-" * 20 + " PPO " + "-" * 20)
    trainer_ppo.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/ppo_", mode)
    print("-" * 20 + " Random Adversary " + "-" * 20)
    trainer_ran.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/ran_", mode)
    print("-" * 20 + " Weighted Random Adversary " + "-" * 20)
    trainer_wro.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/wro_", mode)