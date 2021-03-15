"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""
import sys
import os
import json
import torch
import argparse

from lightsim2grid import LightSimBackend
import grid2op
from grid2op.Action import PlayableAction,TopologyChangeAndDispatchAction
from grid2op.Reward import CombinedScaledReward, L2RPNSandBoxScore, L2RPNReward, GameplayReward
from kaist_agent.Kaist import Kaist

from ppo.ppo import PPO
from ppo.nnpytorch import FFN


def train(env, agent, state_mean, state_std, hyperparameters, actor_model, critic_model):
    """
        Trains the model.
        Parameters:
            env - the environment to train on
            hyperparameters - a dict of hyperparameters to use, defined in main
            actor_model - the actor model to load in if we want to continue training
            critic_model - the critic model to load in if we want to continue training
        Return:
            None
    """ 
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(env=env, agent=agent, policy_class=FFN, state_mean=state_mean, state_std=state_std, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)

def get_args():
    """
        Description:
        Parses arguments at command line.
        Parameters:
            None
        Return:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename

    args = parser.parse_args()

    return args


def main(args):
    """
        The main function to run.
        Parameters:
            args - the arguments parsed from command line
        Return:
            None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    

    # Environment
    backend = LightSimBackend()
    env_name = 'l2rpn_wcci_2020'
    env = grid2op.make(env_name, reward_class=CombinedScaledReward, backend=backend)

    # Agent 
    agent_name = "kaist"
    data_dir = os.path.join('kaist_agent/data')
    with open(os.path.join(data_dir, 'param.json'), 'r', encoding='utf-8') as f:
        param = json.load(f)
    print(param)
    state_mean = torch.load(os.path.join(data_dir, 'mean.pt'), map_location=param['device']).cpu()
    state_std = torch.load(os.path.join(data_dir, 'std.pt'), map_location=param['device']).cpu()
    state_std = state_std.masked_fill(state_std<1e-5, 1.)
    state_mean[0, sum(env.observation_space.shape[:20]):] = 0
    state_std[0, sum(env.observation_space.shape[:20]):] = 1
    agent = Kaist(env, state_mean, state_std, name=agent_name, **param)
    agent.sim_trial = 0
    agent.load_model(data_dir)

    hyperparameters = {
                'timesteps_per_batch': 2048, 
                'max_timesteps_per_episode': 200, 
                'gamma': 0.99, 
                'n_updates_per_iteration': 10,
                'lr': 3e-4, 
                'clip': 0.2,
                'lines_attacked': ['0_4_2', '10_11_11', '11_12_13', '12_13_14', '12_16_20', 
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
                '7_9_9', '8_9_10', '9_16_18', '9_16_19'],
                'attack_duration': 10,
                'danger': 0.9
              }

    # Train or test, depending on the mode specified
    train(env=env, agent=agent, state_mean=state_mean, state_std=state_std, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)

if __name__ == '__main__':
    args = get_args() # Parse arguments from command line
    main(args)