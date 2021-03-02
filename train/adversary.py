import os
import json
import math
import numpy as np
import tensorflow as tf
import grid2op

from grid2op.Opponent import BaseOpponent
from grid2op.Exceptions import OpponentError

# import the train function and train your agent
from l2rpn_baselines.utils import NNParam, TrainingParam
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
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


class D3QN_Opponent(BaseOpponent):
    """
    This opponent will disconnect lines by learning Q-values of attacks among the attackable lines `lines_attacked`.
    When an attack becomes possible, the time of the attack will be sampled uniformly
    in the next `attack_period` steps (see init).
    """

    def __init__(self, action_space, observation_space, lines_attacked=[], attack_period=12*24, name=__name__, is_training=False, learning_rate=cfg.LR):
        BaseOpponent.__init__(self, action_space)

        if len(lines_attacked) == 0:
            warnings.warn(f'The opponent is deactivated as there is no information as to which line to attack. '
                          f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                          f' the opponent to attack in the "make" function.')

        # Store attackable lines IDs
        self._lines_ids = []
        for l_name in lines_attacked:
            l_id = np.where(self.action_space.name_line == l_name)
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError("Unable to find the powerline named \"{}\" on the grid. For "
                                    "information, powerlines on the grid are : {}"
                                    "".format(l_name, sorted(self.action_space.name_line)))

        # Pre-build attacks actions
        self._do_nothing = self.action_space({})
        self._attacks = []
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
        self._attacks = np.array(self._attacks)

        # Opponent's attack period
        self._attack_period = attack_period   

        self.obs_space = observation_space
        
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

        # Compute dimensions from intial spaces
        self.observation_size = self.obs_space.size_obs()
        self.action_size = len(self._attacks)

        # Load network graph
        self.policy_net = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=cfg.LR_DECAY_STEPS,
                                         learning_rate_decay_rate=cfg.LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def _init_training(self):
        self.epsilon = cfg.INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(cfg.PER_CAPACITY, cfg.PER_ALPHA)
        self.target_net = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)            


    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None

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
            The attack performed by the opponent. In this case, a do nothing, all the time.
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        # During creation of the environment, do not attack
        if observation is None:
            return None

        # Register current state to stacking buffer
        self._save_current_frame(observation)
        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return None # Do nothing

        # Decide the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = 1 + self.space_prng.randint(self._attack_period)
        self._next_attack_time -= 1

        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None

        # If all attackable lines are disconnected, do not attack
        status = observation.line_status[self._lines_ids]
        if np.all(status == False):
            return None

        # Infer with the last num_frames states
        a, _ = self.policy_net.predict_move(status, np.array(self.frames))
        return a


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

    def _save_current_frame(self, obs):
        frame = self.convert_obs(obs)
        self.frames.append(frame)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_obs):
        frame = self.convert_obs(next_obs)
        self.frames2.append(frame)
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
    def convert_obs(self, observation):
        li_vect=  []
        for el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)

    ## Baseline Interface
    def reset(self, initial_budget):
        #self._reset_state(observation)
        self._next_attack_time = None
        self._reset_frame_buffer()
    
    def load(self, path):
        self.policy_net.load_network(path)
        if self.is_training:
            self.policy_net.update_target_hard(self.target_net.model)

    def save(self, path):
        self.policy_net.save_network(path)
