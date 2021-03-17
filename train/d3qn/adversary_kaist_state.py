import os
import json
import math
import numpy as np
import tensorflow as tf
import grid2op

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .baseopp import BaseOpponent
from .neuralnet import D3QN

from grid2op.Exceptions import OpponentError

# import the train function and train your agent
from l2rpn_baselines.utils import NNParam, TrainingParam
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
#from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from l2rpn_baselines.DoubleDuelingDQN.prioritized_replay_buffer import PrioritizedReplayBuffer

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import warnings
import numpy as np


class D3QN_Kaist_State_Opponent(BaseOpponent):
    """
    This opponent will disconnect lines by learning Q-values of attacks among the attackable lines `lines_attacked`.
    When an attack becomes possible, the time of the attack will be sampled uniformly
    in the next `attack_period` steps (see init).
    """

    def __init__(self, env, state_mean, state_std, lines_attacked=[], 
                 attack_period=12*24, attack_duration=10, danger=0.9,
                name=__name__, is_training=False, learning_rate=cfg.LR,
                initial_epsilon=cfg.INITIAL_EPSILON,
                final_epsilon=cfg.FINAL_EPSILON,
                decay_epsilon=cfg.DECAY_EPSILON):

        BaseOpponent.__init__(self, env.action_space)

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        self.danger = danger
        # Opponent's attack period
        self.attack_duration = attack_duration
        self._attack_period = attack_period   
        self._next_attack_time = None
        
        # Store constructor params
        self.name = name
        self.num_frames = cfg.N_FRAMES
        self.is_training = is_training
        self.batch_size = cfg.BATCH_SIZE
        self.lr = learning_rate
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []

        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0
        
        self._init_attacks(lines_attacked)
        self._init_obs_converter(state_mean, state_std)

        # Compute dimensions from intial spaces
        self.observation_size = self.observation_space.size_obs()
        self.action_size = len(self._attacks)

        # Load network graph
        self.policy_net = D3QN(self.action_size,
                               self.action_space.dim_topo * 6,          
                               #self.observation_size,
                               num_frames=self.num_frames,
                               learning_rate=self.lr,
                               learning_rate_decay_steps=cfg.LR_DECAY_STEPS,
                               learning_rate_decay_rate=cfg.LR_DECAY_RATE)

        # Setup training vars if needed
        if self.is_training:
            self._init_training()
            
    def _init_attacks(self, lines_attacked):
        if len(lines_attacked) == 0:
            warnings.warn(f'The opponent is deactivated as there is no information as to which line to attack. '
                          f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                          f' the opponent to attack in the "make" function.')

        # Store attackable lines IDs
        self._lines_ids = []
        print(self.action_space.name_line)
        for l_name in lines_attacked:
            l_id = np.where(self.action_space.name_line == l_name)
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError("Unable to find the powerline named \"{}\" on the grid. For "
                                    "information, powerlines on the grid are : {}"
                                    "".format(l_name, sorted(self.action_space.name_line)))

        # Pre-build attacks actions
        self._attacks = []
        self.action2line = {}
        self._attacks.append(self.action_space({}))
        self.action2line[0] = -1
        count = 1
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
            self.action2line[count] = l_id
            count += 1
        self._attacks = np.array(self._attacks)
        self.remaining_time = 0
        self.attack_line = -1
        
    def _init_obs_converter(self, state_mean, state_std):    
        state_std = state_std.masked_fill(state_std < 1e-5, 1.)
        state_mean[0, sum(self.observation_space.shape[:20]):] = 0
        state_std[0, sum(self.observation_space.shape[:20]):] = 1
        self.state_mean = state_mean
        self.state_std = state_std

        self.thermal_limit_under400 = torch.from_numpy(self.env._thermal_limit_a < 400)    
        self.idx = self.observation_space.shape
        self.pp = np.arange(sum(self.idx[:6]),sum(self.idx[:7]))
        self.lp = np.arange(sum(self.idx[:9]),sum(self.idx[:10]))
        self.op = np.arange(sum(self.idx[:12]),sum(self.idx[:13]))
        self.ep = np.arange(sum(self.idx[:16]),sum(self.idx[:17]))
        self.rho = np.arange(sum(self.idx[:20]),sum(self.idx[:21]))
        self.topo = np.arange(sum(self.idx[:23]),sum(self.idx[:24]))
        self.main = np.arange(sum(self.idx[:26]),sum(self.idx[:27]))
        self.over = np.arange(sum(self.idx[:22]),sum(self.idx[:23]))
        
        # parse substation info
        self.subs = [{'e':[], 'o':[], 'g':[], 'l':[]} for _ in range(self.action_space.n_sub)]
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid):
            self.subs[sub_id]['g'].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid):
            self.subs[sub_id]['l'].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid):
            self.subs[sub_id]['o'].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid):
            self.subs[sub_id]['e'].append(ex_id)
        
        self.sub_to_topos = []  # [0]: [0, 1, 2], [1]: [3, 4, 5, 6, 7, 8]
        for sub_info in self.subs:
            a = []
            for i in sub_info['e']:
                a.append(self.action_space.line_ex_pos_topo_vect[i])
            for i in sub_info['o']:
                a.append(self.action_space.line_or_pos_topo_vect[i])
            for i in sub_info['g']:
                a.append(self.action_space.gen_pos_topo_vect[i])
            for i in sub_info['l']:
                a.append(self.action_space.load_pos_topo_vect[i])
            self.sub_to_topos.append(torch.LongTensor(a))

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)
        self.max_n_line = max([len(topo['o'] + topo['e']) for topo in self.subs])
        self.max_n_or = max([len(topo['o']) for topo in self.subs])
        self.max_n_ex = max([len(topo['e']) for topo in self.subs])
        self.max_n_g = max([len(topo['g']) for topo in self.subs])
        self.max_n_l = max([len(topo['l']) for topo in self.subs])
        self.n_feature = 6
        

    def _init_training(self):
        self.epsilon = cfg.INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(cfg.PER_CAPACITY, cfg.PER_ALPHA)
        self.target_net = D3QN(self.action_size,
                               self.action_space.dim_topo * 6,
                                #self.observation_size,
                                num_frames = self.num_frames)            


    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None
        
    def take_step(self, observation, agent_action=None, env_action=None, budget=None):
        self._save_current_frame(self.convert_obs(observation))

    def attack(self, observation, agent_action=None, env_action=None, budget=None, previous_fails=False):
        """
        This method is the equivalent of "act" for a regular agent.
        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.
        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)
        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took
        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.
        budget: ``float``
            The current remaining budget (if an action is above this budget, it will be replaced by a do nothing.
        previous_fails: ``bool``
            Wheter the previous attack failed (due to budget or ambiguous action)
        Returns
        -------
        attack: :class:`grid2op.Action.Action`
            The attack performed by the opponent. 
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        # During creation of the environment, do not attack
        if observation is None:
            return self._do_nothing, 0

        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return self._do_nothing, 0

        # Decide the time of the next attack
        '''
        if self._next_attack_time is None:
            self._next_attack_time = 1 + self.space_prng.randint(self._attack_period)
        self._next_attack_time -= 1
        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return self._do_nothing, 0
        '''

        # Get attackable lines
        status = observation.line_status[self._lines_ids]
        status = np.insert(status, 0, True, axis=0) # do nothing is always valid

        # Epsilon variation
        if np.random.rand(1) < self.epsilon and self.is_training:
            # TODO: use random move
            if np.all(~status): # no available line to attack (almost 0 probability)
                return None, None
            a = np.random.randint(0, len(self._attacks) - 1)
            while not status[a]: # repeat util line of status True is chosen
                a = np.random.randint(0, len(self._attacks) - 1)
            self.remaining_time = self.attack_duration
            return (self._attacks[a], a) 
        else:
            # Infer with the last num_frames states
            a, _ = self.policy_net.predict_move(status, np.array(self.frames))
            self.remaining_time = self.attack_duration
            self.attack_line = self.action2line[a]
            return (self._attacks[a], a)       


    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = cfg.DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = cfg.INITIAL_EPSILON * -math.log10((step_off + 1) / (cfg.DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(cfg.INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(cfg.FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip
            
    def _save_hyperparameters(self, logpath, env, steps):
        try:
            # change of name in grid2op >= 1.2.3
            r_instance = env._reward_helper.template_reward
        except AttributeError as nm_exc_:
            r_instance = env.reward_helper.template_reward
        hp = {
            "lr": cfg.LR,
            "lr_decay_steps": cfg.LR_DECAY_STEPS,
            "lr_decay_rate": cfg.LR_DECAY_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "stack_frames": cfg.N_FRAMES,
            "iter": steps,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "per_alpha": cfg.PER_ALPHA,
            "per_beta": cfg.PER_BETA,
            "per_capacity": cfg.PER_CAPACITY,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def state_normalize(self, s):
        s = (s - self.state_mean) / self.state_std
        return s
    
    
    def convert_obs(self, o):
        # o.shape : (B, O)
        # output (Batch, Node, Feature)
        o = torch.FloatTensor(o.to_vect()).unsqueeze(0)
        o = self.state_normalize(o)

        length = self.action_space.dim_topo # N
        p_ = torch.zeros(o.size(0), length).to(o.device)    # (B, N)
        p_[..., self.action_space.gen_pos_topo_vect] = o[...,  self.pp]
        p_[..., self.action_space.load_pos_topo_vect] = o[..., self.lp]
        p_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.op]
        p_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.ep]

        rho_ = torch.zeros(o.size(0), length).to(o.device)
        rho_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.rho]
        rho_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.rho]

        danger_ = torch.zeros(o.size(0), length).to(o.device)
        danger = ((o[...,self.rho] >= self.danger-0.05) & self.thermal_limit_under400.to(o.device)) | (o[...,self.rho] >= self.danger)
        danger_[..., self.action_space.line_or_pos_topo_vect] = danger.float()
        danger_[..., self.action_space.line_ex_pos_topo_vect] = danger.float()      

        over_ = torch.zeros(o.size(0), length).to(o.device)
        over_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.over]/3
        over_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.over]/3

        main_ = torch.zeros(o.size(0), length).to(o.device)
        temp = torch.zeros_like(o[..., self.main])
        temp[o[..., self.main] ==0] = 1
        main_[..., self.action_space.line_or_pos_topo_vect] = temp
        main_[..., self.action_space.line_ex_pos_topo_vect] = temp

        topo_ = o[..., self.topo]
        state = torch.stack([p_, rho_, danger_, topo_, over_, main_], dim=2) # B, N, F
        state = torch.reshape(state, (state.size(0), -1))
        # print(state.numpy().shape) # (1, 1062)
        return state.numpy()

    ## Baseline Interface
    def reset(self, obs):
        self._reset_state(obs)
        self._next_attack_time = None
        self._reset_frame_buffer()
    
    def load(self, path):
        self.policy_net.load_network(path)
        if self.is_training:
            self.policy_net.update_target_hard(self.target_net.model)

    def save(self, path):
        self.policy_net.save_network(path)

    def _batch_train(self, training_step, step):
        """Trains network to fit given parameters"""

        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, cfg.PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.action_space.dim_topo * 6 * self.num_frames
        #print(input_size)
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.policy_net.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.policy_net.model.predict(input_t_1, batch_size=self.batch_size)
        Q2 = self.target_net.model.predict(input_t_1, batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.policy_net.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.policy_net.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)

        # Log some useful metrics every even updates
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.policy_net.train_lr, step)
            if cfg.VERBOSE:
                print("loss =", loss)