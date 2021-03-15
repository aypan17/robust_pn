'''
Pytorch version of the Grid2Op D3QN neural network.
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FFN(nn.Module):
	"""
		A standard Feed Forward Neural Network.
	"""
	def __init__(self, observation_size, action_size, model_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				observation_size - output dimensions as an int
				action_size - input dimensions as an int
			Return:
				None
		"""
		super(FFN, self).__init__()

		self.in_dim = observation_size
		self.out_dim = action_size
		self.model_dim = model_dim

		self.layer1 = nn.Linear(self.in_dim, self.model_dim)
		self.layer2 = nn.Linear(self.model_dim, self.model_dim)
		self.layer3 = nn.Linear(self.model_dim, self.out_dim)

	def forward(self, state):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				state - converted observation to pass as input
			Return:
				out - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array

		x = F.relu(self.layer1(state))
		x = F.relu(self.layer2(x))
		out = self.layer3(x)

		return out

	