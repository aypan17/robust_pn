import os
import json
import math
import numpy as np
import tensorflow as tf
import torch

from matplotlib import animation
import matplotlib.pyplot as plt
from IPython import display

import grid2op
from grid2op import make
from grid2op.Runner import Runner
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward

from kaist_agent.Kaist import Kaist

from simple_opponents.random_opponent import RandomOpponent, WeightedRandomOpponent
from d3qn.adversary import D3QN_Opponent
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg

from ppo.ppo import PPO
from ppo.nnpytorch import FFN


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def evaluate(env, agent, opponent, n_episodes, max_steps, verbose=False):
    reward_arr, n_survive_steps_arr = [], []
    for i_episode in range(1, n_episodes+1):
        step = 0
        obs = env.reset()
        agent.reset(obs)
        total_reward = 0
        frames = []

        while step < max_steps:
            frames.append(env.render())
            #display.clear_output(wait=True)
            #display.display(plt.gcf())

            # agent act
            a = agent.act(obs, None, None)
            obs, reward, done, info = env.step(a)
            
            total_reward += reward
            if done:
                break

            # opponent attack
            if opponent:
                if opponent.remaining_time >= 0:
                    obs.time_before_cooldown_line[opponent.attack_line] = opponent.remaining_time
                    opponent.remaining_time -= 1
                else:
                    attack = opponent.act(obs, None, None)
                    obs, opp_reward, done, info = env.step(attack) 

            if done:
                break
            step += 1            
            
        reward_arr.append(total_reward)
        n_survive_steps_arr.append(step)
        
    if verbose:
        for i in range(1, n_episodes+1):
            print(f'Episode {i}/{n_episodes} - Reward: {reward_arr[i-1]:.2f}\t Number of steps survived: {n_survive_steps_arr[i-1]}')
        
    return reward_arr, n_survive_steps_arr, frames

def main():
    env_name = 'l2rpn_wcci_2020'
    env = make(env_name, reward_class=L2RPNSandBoxScore,
               other_rewards={
                   "reward": L2RPNReward
               })

    agent_name = "kaist"
    data_dir = os.path.join('kaist_agent/data')
    with open(os.path.join(data_dir, 'param.json'), 'r', encoding='utf-8') as f:
        param = json.load(f)

    state_mean = torch.load(os.path.join(data_dir, 'mean.pt'), map_location=param['device']).cpu()
    state_std = torch.load(os.path.join(data_dir, 'std.pt'), map_location=param['device']).cpu()
    state_std = state_std.masked_fill(state_std<1e-5, 1.)
    state_mean[0, sum(env.observation_space.shape[:20]):] = 0
    state_std[0, sum(env.observation_space.shape[:20]):] = 1
    agent = Kaist(env, state_mean, state_std, name=agent_name, **param)
    agent.sim_trial = 0
    agent.load_model(data_dir)

    n_episodes = 1
    n_max_steps = 150

    # opponent hyperparameters
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

    opponent_name = "ppo_untrained"
    opponent = PPO(env=env, agent=agent, policy_class=FFN, state_mean=state_mean, state_std=state_std, **hyperparameters)
    #opponent.actor.load_state_dict(torch.load('./ppo_actor.pth'))

    gif_name = agent_name+"vs"+opponent_name+".gif"

    reward_arr, n_survive_steps_arr, frames = evaluate(env, agent, opponent, n_episodes, n_max_steps, verbose=True)
    print()
    print('Average reward: {:.2f}\t Average number of steps survived: {}'.format(np.mean(reward_arr), np.mean(n_survive_steps_arr)))
    save_frames_as_gif(frames, filename="gif_name")
    print(f"Saved gif of environment in {gif_name}")

if __name__ == '__main__':
    main()