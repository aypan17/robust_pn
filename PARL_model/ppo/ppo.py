import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, env, agent, policy_class, state_mean, state_std, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
            Returns:
                None
        """
        # Make sure the environment is compatible with our code
        #assert(type(env.observation_space) == gym.spaces.Box)
        #assert(type(env.action_space) == gym.spaces.Box)

        # Extract environment information
        self.env = env
        self.agent = agent
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)
        self._init_attacks(hyperparameters)
        self._init_obs_converter(hyperparameters, state_mean, state_std)

        # Set environment variables
        self.act_dim = len(self._attacks)
        self.obs_dim = self.state_dim        

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim, self.model_dim)                                                   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1, self.model_dim)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation: 
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        """
            Too many transformers references, I'm sorry. This is where we collect the batch of data
            from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. Note that obs is short for observation. 
            obs = self.env.reset()
#             self.agent.reset(obs)
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                state = self.convert_obs(obs)
                batch_obs.append(state)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                action, log_prob = self.get_action(state)
                attack = self._attacks[action]
                obs, rew, done, info = self.env.step(attack)

                # Allow agent response:
                while self.remaining_time > 0 and not done:
                    obs.time_before_cooldown_line[self.attack_line] = self.remaining_time
                    response = self.agent.act(obs, None, None)
                    obs, rew, done, info = self.env.step(response)
                    self.remaining_time -= 1

                # Track recent reward, action, and action log probability
                ep_rews.append(-1 * rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.cat(batch_obs, dim=0)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def act(self, obs, reward, done):
        """
            Used to evaluate the learned policy of the actor in grid2op.
        """
        mean = self.actor(self.convert_obs(obs))

        # Create a distribution with the given mean
        dist = Categorical(logits=mean)

        # Sample an action from the distribution
        action = dist.sample().item()

        # Set remaining time of attack
        self.remaining_time = self.attack_duration
        self.attack_line = self.action2line[action]

        return self._attacks[action], action

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = Categorical(logits=mean)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Set remaining time of attack
        self.remaining_time = self.attack_duration
        self.attack_line = self.action2line[action.item()]

        # Return the sampled action and the log probability of that action in our distribution
        return action.item(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        #dist = MultivariateNormal(mean, self.cov_mat)
        #log_probs = dist.log_prob(batch_acts)
        dist = Categorical(logits=mean)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

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
        return state

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200           # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 0.0005                                 # Learning rate of actor optimizer
        self.gamma = 0.99                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.model_dim = 128                            # Hidden size of model
        self.state_dim = 1062                            # Size of model embedding

        # Miscellaneous parameters
        #self.render = True                              # If we should render during rollout
        #self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 1                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _init_attacks(self, hyperparameters):
        lines_attacked = hyperparameters['lines_attacked']
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

        # Pre-build attack actions
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

        self.attack_duration = hyperparameters['attack_duration']
        self.remaining_time = 0
        self.attack_line = -1

    def _init_obs_converter(self, hyperparameters, state_mean, state_std):    
        self.danger = hyperparameters['danger']
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

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.
            Parameters:
                None
            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []