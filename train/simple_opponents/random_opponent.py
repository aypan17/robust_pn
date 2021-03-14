import os
import json
import math
import numpy as np
import tensorflow as tf
import grid2op

from .baseopp import BaseOpponent

class RandomOpponent(BaseOpponent):
    """ This opponent will disconnect lines randomly"""
    def __init__(self, observation_space, action_space, lines_attacked=[], attack_duration=10,
                 attack_period=12*24, name=__name__):
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
        self._attacks = []
        self.action2line = {}
        count = 0
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
            self.action2line[count] = l_id
            count += 1
        self._attacks = np.array(self._attacks)
        
        self._next_attack_time = None
        self._attack_period = attack_period
        
        self.attack_duration = attack_duration
        self.remaining_time = 0
        self.attack_line = -1
        
    def reset(self, initial_budget=None):
        self._next_attack_time = None
        self.remaining_time = 0
        
    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None
    
    def attack(self, observation, agent_action=None, env_action=None, budget=None,
               previous_fails=None):
        if observation is None:  # during creation of the environment
            return None  # i choose not to attack in this case

        # Decide the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = 1 + self.space_prng.randint(self._attack_period)
        self._next_attack_time -= 1
        
        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None
        
        # Status of attackable lines
        status = observation.line_status[self._lines_ids]
        # If all attackable lines are disconnected
        if np.all(~status):
            return None, None  # i choose not to attack in this case

        # Pick a line among the connected lines
        a = np.random.randint(0, len(self._attacks) - 1)
        while not status[a]: # repeat until line of status True is chosen
            a = np.random.randint(0, len(self._attacks) - 1)        
        self.remaining_time = self.attack_duration
        self.attack_line = self.action2line[a]
            
        return self._attacks[a], a
    
class WeightedRandomOpponent(BaseOpponent):
    """ This opponent will disconnect lines randomly"""
    def __init__(self, observation_space, action_space, lines_attacked=[], attack_duration=10,
                 attack_period=12*24, name=__name__):
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
        self._attacks = []
        self.action2line = {}
        count = 0
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
            self.action2line[count] = l_id
            count += 1
        self._attacks = np.array(self._attacks)
        
        self.rho_total = np.zeros_like(lines_attacked)
        self.obs_count = 0
        
        self._next_attack_time = None
        self._attack_period = attack_period
        
        self.attack_duration = attack_duration
        self.remaining_time = 0
        self.attack_line = -1
        
    def reset(self, initial_budget=None):
        self._next_attack_time = None
        self._rho_total = np.zeros_like(self._lines_ids).astype(np.float64)
        self.obs_count = 0
        self.remaining_time = 0
        
    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None
        
    def take_step(self, observation, agent_action=None, env_action=None, budget=None,
                    previous_fails=None):
        self.obs_count += 1
        self._rho_total += observation.rho[self._lines_ids]
    
    def attack(self, observation, agent_action=None, env_action=None, budget=None,
               previous_fails=None):
        if observation is None:  # during creation of the environment
            return None  # i choose not to attack in this case
        
        # Status of attackable lines
        status = observation.line_status[self._lines_ids]
        
        # update rho normalization
        _rho_normalization = self._rho_total / self.obs_count
        
        # Decide the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = 1 + self.space_prng.randint(self._attack_period)
        self._next_attack_time -= 1
        
        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None
        
        # If all attackable lines are disconnected
        if np.all(~status):
            return None  # i choose not to attack in this case

        # Pick a line among the connected lines
        rho_norm_nonzero_ind = np.nonzero(_rho_normalization[status])
        rho = observation.rho[self._lines_ids][status]        
        rho[rho_norm_nonzero_ind] = rho[rho_norm_nonzero_ind] / _rho_normalization[status][rho_norm_nonzero_ind]

        # choose line among lines of True status following the probabilty distribution of rho
        x = np.random.uniform()
        rho_idx = 0 # since rho has length of true status, we need to keep a separate rho_idx
        rho_sum = rho.sum()
        p_cum = 0 # cumulative probability
        for i in range(len(self._attacks)):
            if not status[i]:
                continue
            p_cum += rho[rho_idx] / rho_sum
            if x <= p_cum:
                a = i
                break
            rho_idx += 1
        self.remaining_time = self.attack_duration
        self.attack_line = self.action2line[a]
        
        return self._attacks[a], a