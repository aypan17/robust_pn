import os
import json
import math
import numpy as np
import tensorflow as tf
import torch

import grid2op
from lightsim2grid import LightSimBackend
from adversary import D3QN_Opponent
from grid2op.Agent import DoNothingAgent
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import CombinedScaledReward
#from l2rpn_baselines.Kaist.Kaist import Kaist


MAX_TIMESTEP = 7 * 288

# n_iter, save_path, save_path_opponent, num_pre_training_steps, log_path
def train(env, agent, opponent, n_iter, save_path, save_path_opponent,
          num_pre_training_steps, log_path, epsilon):
    # Make sure we can fill the experience buffer
    if num_pre_training_steps < agent.batch_size * agent.num_frames:
        num_pre_training_steps = agent.batch_size * agent.num_frames
        
    # Loop vars
    num_training_steps = hparam.n_iter
    num_steps = num_pre_training_steps + num_training_steps
    step = 0
    opponent.epsilon = epsilon
    alive_steps = 0
    total_reward, total_reward_opponent = 0, 0
    agent.done, opponent.done = True, True
    print(f"Total number of steps: {num_steps}")

    # Create file system related vars
    logpath = os.path.join(log_path, agent.name)
    opponent_logpath = os.path.join(log_path, opponent.name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_opponent, exist_ok=True)
    modelpath = os.path.join(save_path, agent.name + ".h5")
    opponent_modelpath = os.path.join(save_path_opponent, opponent.name + '.h5')
    agent.tf_writer = tf.summary.create_file_writer(logpath, name=agent.name)
    opponent.tf_writer = tf.summary.create_file_writer(opponent_logpath, name=opponent.name)
    opponent._save_hyperparameters(save_path_opponent, env, num_steps)

    steps_buffer = []
    rewards_buffer = []
    experience_buffer = []
    print("hi")
    for _ in range(num_episodes):
        _ = self.env.reset()
        max_day = (
            self.env.chronics_handler.max_timestep() - MAX_TIMESTEP) // 288
        start_timestep = np.random.randint(
            max_day) * 288 - 1  # start at 00:00
        if start_timestep > 0:
            self.env.fast_forward_chronics(start_timestep)

        steps = 0
        obs = self.env.get_obs()
        done = False

        while not done:
            # Execute attack
            opponent._save_current_frame(obs)
            attack = opponent.attack(obs)
            attack_obs, pre_reward, _, info = env.step(attack)
            assert not info['is_illegal'] and not info['is_ambiguous']

            # Execute agent response
            action = self.agent.act(attack_obs, None, None)
            obs, reward, done, info = self.env.step(action)
            assert not info['is_illegal'] and not info['is_ambiguous']

            rewards += -1 * reward
            steps += 1
            if steps >= MAX_TIMESTEP:
                break

            # Perform training when we have enough experience in buffer
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Decay chance of random action
                opponent.epsilon = opponent._adaptive_epsilon_decay(training_step)

                # Perform training at given frequency
                if step % cfg.UPDATE_FREQ == 0 and \
                   len(opponent.per_buffer) >= opponent.batch_size:
                    # Perform training
                    opponent._batch_train(training_step, step)

                    if cfg.UPDATE_TARGET_SOFT_TAU > 0.0:
                        tau = cfg.UPDATE_TARGET_SOFT_TAU
                        # Update target network towards primary network
                        opponent.policy_net.update_target_soft(opponent.target_net.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    opponent.policy_net.update_target_hard(opponent.target_net.model)

            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                agent.save(modelpath)
                opponent.save(modelpath_opponent)

        steps_buffer.append(steps)
        rewards_buffer.append(rewards)

        

    # Train the opponent
    # Store the episodes
    # monte carlo policy gradient for the agent
    # (state, action, reward) --- idk how to do this

    # Save model after all steps
    agent.save(modelpath)
    opponent.save(modelpath_opponent)
    

def main():

    backend = LightSimBackend()
    env_name = 'l2rpn_wcci_2020'

    env = grid2op.make(env_name,
               action_class=TopologyChangeAndDispatchAction,
               reward_class=CombinedScaledReward,
               backend=backend)

    opp_name = "D3QN Opponent"
    learning_rate = 0.001
    print(env.action_space)
    print(env)
    opp = D3QN_Opponent(env, env.action_space, env.observation_space, name=opp_name,
                         is_training=True, learning_rate=learning_rate)

    data_dir = os.path.join('.', 'data')
    with open(os.path.join(data_dir, 'param.json'), 'r', encoding='utf-8') as f:
        param = json.load(f)

    # Create the agent
    '''
    state_mean = torch.load(os.path.join(data_dir, 'mean.pt'), map_location=param['device']).cpu()
    state_std = torch.load(os.path.join(data_dir, 'std.pt'), map_location=param['device']).cpu()
    state_std = state_std.masked_fill(state_std<1e-5, 1.)
    state_mean[0, sum(env.observation_space.shape[:20]):] = 0
    state_std[0, sum(env.observation_space.shape[:20]):] = 1
    agent = Kaist(env, state_mean, state_std, **param)
    agent.load_model(data_dir)
    '''
    agent = DoNothingAgent()
    n_iter = 1000
    save_path = "agent"
    save_path_opponent = "opp"
    num_pre_training_steps = 10
    log_path = "log"
    epsilon = 0.1
    train(env, agent, opponent, n_iter, save_path, save_path_opponent,
          num_pre_training_steps, log_path, epsilon)

if __name__ == '__main__':
    main()