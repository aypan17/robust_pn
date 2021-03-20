import os
import json
import math
import numpy as np
import tensorflow as tf
import torch

import grid2op
from d3qn.adversary import D3QN_Opponent
from d3qn.adversary_mixed_state import D3QN_Mixed_State_Opponent
from grid2op.Agent import DoNothingAgent
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import CombinedScaledReward, L2RPNSandBoxScore, L2RPNReward, GameplayReward
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg

from kaist_agent.Kaist import Kaist

MAX_TIMESTEP = 7 * 288

def train_adversary(env, agent, opponent, num_pre_training_steps, n_iter, save_path, log_path):
    # Make sure we can fill the experience buffer
    if num_pre_training_steps < opponent.batch_size * opponent.num_frames:
        num_pre_training_steps = opponent.batch_size * opponent.num_frames
        
    # Loop vars
    num_training_steps = n_iter
    num_steps = num_pre_training_steps + num_training_steps
    step = 0
    alive_steps = 0
    total_reward = 0
    done = True
    print(f"Total number of steps: {num_steps}")

    # Create file system related vars
    logpath = os.path.join(log_path, opponent.name)
    os.makedirs(save_path, exist_ok=True)
    modelpath = os.path.join(save_path, opponent.name + ".h5")
    opponent.tf_writer = tf.summary.create_file_writer(logpath, name=opponent.name)
    opponent._save_hyperparameters(save_path, env, num_steps)
    
    while step < num_steps:
        # Init first time or new episode
        if done:
            new_obs = env.reset() # This shouldn't raise
            agent.reset(new_obs)
            opponent.reset(new_obs)
            done = False
        if cfg.VERBOSE and step % 1000 == 0:
            print("Step [{}] -- Random [{}]".format(step, opponent.epsilon))

        # Save current observation to stacking buffer
        opponent._save_current_frame(opponent.def_state, opponent.kaist_state)

        # Execute attack if allowed
        if step <= num_pre_training_steps:
            opponent.remaining_time = 0
            attack, a = opponent._do_nothing, 0
        else:
            attack, a = opponent.attack(new_obs)

        if a != 0:
#             print(f'ATTACK step {step}: disconnected {a}')
            attack_obs, opp_reward, done, info = env.step(attack)
            if info["is_illegal"] or info["is_ambiguous"] or \
               info["is_dispatching_illegal"] or info["is_illegal_reco"]:
                if cfg.VERBOSE:
                    print(attack, info)
            new_obs = attack_obs
            opponent.tell_attack_continues(None, None, None, None)

        while opponent.remaining_time >= 0 and not done:
            new_obs.time_before_cooldown_line[opponent.attack_line] = opponent.remaining_time
            response = agent.act(new_obs, None, None)
            new_obs, reward, done, info = env.step(response)
            opponent.remaining_time -= 1
            total_reward += reward
            alive_steps += 1
        
        # Save new observation to stacking buffer
        new_state_def, new_state_kaist = opponent.convert_obs(new_obs)
        opponent._save_next_frame(new_state_def, new_state_kaist)

        # Save to experience buffer
        if len(opponent.def_frames2) == opponent.num_frames:
            opponent.per_buffer.add((np.array(opponent.def_frames), np.array(opponent.kaist_frames)),
                                a, -1 * reward,
                                (np.array(opponent.def_frames2), np.array(opponent.kaist_frames2)),
                                opponent.done)


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
        
        if done:
            opponent.epoch_rewards.append(-1 * total_reward)
            opponent.epoch_alive.append(alive_steps)
            if cfg.VERBOSE and step > num_pre_training_steps:
                print("step {}: Agent survived [{}] steps with reward {}".format(step, alive_steps, total_reward))
            alive_steps = 0
            total_reward = 0         
        else:
            alive_steps += 1
            
        ######## After Each Step #######
        if step > 0 and step % 2000 == 0: # save network every 5000 iters
            opponent.save(modelpath)
        step += 1
        # Make new obs the current obs
        opponent.obs = new_obs
        opponent.def_state = new_state_def
        opponent.kaist_state = new_state_kaist

    # Save model after all steps
    opponent.save(modelpath)


def main():
    env_name = 'l2rpn_wcci_2020'
    env = grid2op.make(env_name, reward_class=CombinedScaledReward)

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

    # Opponent 
    opponent_name = "D3QN_kaist_state"
    num_pre_training_steps = 256
    learning_rate = 5e-5
    initial_epsilon = 0.99
    final_epsilon = 0.01
    decay_epsilon = 15000
    attack_period = 20
    lines = ['0_4_2', '10_11_11', '11_12_13', '12_13_14', '12_16_20', 
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

    opponent = D3QN_Mixed_State_Opponent(env, state_mean, state_std,
                             lines_attacked=lines, attack_period=attack_period,
                             name=opponent_name, is_training=True, value_is_def=True, learning_rate=learning_rate,
                             initial_epsilon=initial_epsilon, final_epsilon=final_epsilon, decay_epsilon=decay_epsilon)

    # Training
    n_iter = 15000
    # Register custom reward for training
    cr = env._reward_helper.template_reward
    #cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    #cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0/float(env.n_line))
    # Initialize custom rewards
    cr.initialize(env)
    # Set reward range to something managable
    cr.set_range(-1.0, 1.0)

    save_path = "kaist_agent_D3QN_mixed_state_opponent_{}_{}".format(attack_period, n_iter)
    log_path="tf_logs_D3QN"

    train_adversary(env, agent, opponent, num_pre_training_steps, n_iter, save_path, log_path)

if __name__ == '__main__':
    main()