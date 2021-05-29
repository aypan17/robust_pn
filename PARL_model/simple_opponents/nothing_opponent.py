import os
import json
import math
import numpy as np
import tensorflow as tf
import grid2op
import warnings

from .baseopp import BaseOpponent

class DoNothingOpponent(BaseOpponent):
    def __init__(self, observation_space, action_space):
        BaseOpponent.__init__(self, action_space)
        self.remaining_time = -1
        
    def attack(self, observation):
        return None