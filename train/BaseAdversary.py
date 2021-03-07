import os
import json
import math
import numpy as np
import tensorflow as tf
import grid2op

from baseopp import BaseOpponent
from grid2op.Exceptions import OpponentError

class BaseAdversary(BaseOpponent)
    """
    An adversary that learns which lines to attack. Agnostic to PyTorch/TF.
    
    Input: state representation, attack algorithm (TODO), training (TODO)

    Returns: grid2op.Opponent object
    """
    def __init__(self, env, state_params, attack_params, train_params):
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.init_state(state_params)
        self.init_attack(attack_params)
        self.init_train(train_params)

    def init_state(self, state_params):
        """
        A dictionary of state representation hyperparameters.

        State params must contain 'convert' key which holds the converter.
        """
        self.convert = state_params['convert']
        self.max_frames = state_params['max_frames']
        self.state_buffer = []
        self.state = None
        self.state_params = state_params


    def init_attack(self, attack_params):
        self.attack_params = attack_params

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

        # Pre-build attacks and make do_nothing an option
        self._attacks = [self._do_nothing]
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
        self._attacks = np.array(self._attacks)

        self.next_attack_time = None

    def init_train(self, train_params):
        self.train_params = train_params
    

    def reset(self, initial_budget):
        self.state_buffer = []
        self.state = None

    def get_state(self):
        return self.state

    def set_state(self, obs):
        self.state = self.convert(obs)

    def stack_obs(self, obs):
        state = self.convert_obs(obs)
        if len(self.state_buffer) == 0:
            for _ in range(self.max_frames):                
                self.state_buffer.append(state)
        else:
            self.state_buffer.pop(0)
            self.state_buffer.append(state)

    def convert_obs(self, obs):
        return self.convert(obs)

    def _save_hyperparameters(self, logpath, env, steps):
        try:
            # change of name in grid2op >= 1.2.3
            r_instance = env._reward_helper.template_reward
        except AttributeError as nm_exc_:
            r_instance = env.reward_helper.template_reward
        
        hp = {'state': self.state_params, 'train': self.train_params, 'attack': self.attack_params}
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    def tell_attack_continues(self, obs, agent_act, env_act, budget):
        """
        The purpose of this method is to tell the agent that his attack is being continued
        and to indicate the current state of the grid.
        
        At every time step, either "attack" or "tell_acttack_continues" is called exactly once.
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
        """
        self.stack_obs(obs)
        self.next_attack_time = None

    def attack(self, obs):
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
        self.stack_obs(obs)

        # Get attackable lines
        status = np.concatenate(([True], observation.line_status[self._lines_ids]))

        # Epsilon variation
        if np.random.rand(1) < self.epsilon:
            # Use random attack
            a, _ = self.random_attack(status)
        else:
            # Infer with the last num_frames states
            a, _ = self.predict_attack(status)
        return (self._attacks[a], a)      

    def random_attack(self, status):
        """
        Perform a random attack given online lines.

        Inputs:
            status: A boolean list of the lines which are currently available to be disconnected.

        Return:
            attack: A grid2op.Attack object
        """
        pass

    def predict_attack(self, status):
        """
        Perform a learned attack given online lines.

        Inputs:
            status: A boolean list of the lines which are currently available to be disconnected.

        Return:
            attack: A grid2op.Attack object
        """
        pass


