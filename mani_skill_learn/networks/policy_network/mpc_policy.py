from mani_skill_learn.utils.torch.module_utils import CustomDataParallel
from ..builder import POLICYNETWORKS
from mani_skill_learn.utils.data import to_torch, to_np, squeeze
from mani_skill_learn.utils.data.concat import stack_list_of_array as stack
from mani_skill_learn.networks.model_network import PointNetModel
import copy, numpy as np
import torch

# Adapted from the homework 4 of cs285 UC Berkeley (2021 fall), which is completed by 
# Deming Chen (Email: cdm@pku.edu.cn). Original code of the homework:
# https://github.com/berkeleydeeprlcourse/homework_fall2021/tree/main/hw4/cs285/models
@POLICYNETWORKS.register_module()
class MPCPolicy():
    def __init__(self,                 
                 ac_dim,
                 ac_space,
                 horizon,
                 num_action_sequences,
                 cem_cfg,
                 mppi_cfg,
                 noise_std=0.1,
                 num_ensemble=1,
                 model=None,
                 evaluator=None,
                 sample_strategy='random'
                 ):

        # init vars
        self.horizon = horizon
        self.N = num_action_sequences
        self.noise_std = noise_std
        self.num_ensemble = num_ensemble
        self.model = model
        self.evaluator = evaluator

        # action space
        self.ac_dim = ac_dim
        self.ac_space = ac_space
        self.low = self.ac_space.low
        self.high = self.ac_space.high
        
        # Sampling strategy
        allowed_sampling = ('random', 'cem', 'mppi')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        if sample_strategy == 'cem':
            self.cem_iterations = cem_cfg['cem_iterations']
            self.cem_num_elites = cem_cfg['cem_num_elites']
            self.cem_alpha = cem_cfg['cem_alpha']
        elif sample_strategy == 'mppi':
            self.mppi_gamma = mppi_cfg['mppi_gamma']
            self.mppi_beta = mppi_cfg['mppi_beta']
            self.sample_velocity = mppi_cfg['sample_velocity']
            self.mppi_sigma = mppi_cfg['mag_noise'] * (self.high - self.low) / 2 # Shape: (ac_dim,)
            self.mppi_mean = None # np.tile((self.high + self.low) / 2, (self.horizon, 1))

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")
        if self.sample_strategy == 'mppi':
            print(f"MPPI params: gamma={self.mppi_gamma}, beta={self.mppi_beta}, sigma={self.mppi_sigma}")
    
    @property
    def device(self):
        if isinstance(self.model, PointNetModel) or isinstance(self.model, CustomDataParallel):
            return self.model.device
        else:
            raise NotImplementedError

    def reset(self, num_procs=1, which=-1, **reset_kwargs):
        if self.sample_strategy == 'mppi':
            if self.mppi_mean is None or num_procs != self.mppi_mean.shape[0] or which == -1:
                self.mppi_mean = np.tile((self.high + self.low) / 2, (num_procs, self.horizon, 1))
            else:
                self.mppi_mean[which] == np.tile((self.high + self.low) / 2, (self.horizon, 1))

    def head(self, action, num_actions=1):
        """
        Input:
            one action of shape (1, ac_dim)
        Return:
            noisy action with exploration,
            log p,
            mean action, 
            log std, 
            std.
        """
        assert num_actions == 1
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=action.shape)
        noisy_ac = np.clip(action + noise, self.low, self.high)
        return noisy_ac, np.ones_like(action) * -np.inf, action, np.ones_like(action) * -np.inf, np.zeros_like(action)

    #get_action
    def __call__(self, obs, num_actions=1, mode='sample'):
        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[1] == 1:
            # CEM: only a single action sequence to consider; return the first action
            # Shape of candidate_action_sequences: (num_obs, 1, horizon, ac_dim)
            action_to_take = candidate_action_sequences[:, 0, 0]
        else: # Random
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)] # TODO (Q2)
            action_to_take = best_action_sequence[0][None]  # TODO (Q2)
        all_info = self.head(action_to_take, num_actions)
        sample, log_prob, mean = all_info[:3]
        if mode == 'all':
            return all_info
        elif mode == 'eval':
            return mean
        elif mode == 'sample':
            return sample
        else:
            raise ValueError(f"Unsupported mode {mode}!!")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if isinstance(obs, dict): # mode = 'pointcloud'
            num_obs = obs['state'].shape[0]
        elif isinstance(obs, np.ndarray): # mode = 'state'
            num_obs = obs.shape[0]
        else: num_obs = 1
        assert num_obs == self.mppi_mean.shape[0]

        if self.sample_strategy == 'random' or ((self.sample_strategy == 'cem' or self.sample_strategy == 'mppi') 
            and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            
            random_action_sequences = np.random.uniform(self.low, self.high, \
                                                        (num_obs, num_sequences, horizon, self.ac_dim))
            return random_action_sequences

        elif self.sample_strategy == 'cem':
            raise NotImplementedError
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            candidate_sequences = np.random.uniform(self.low, self.high, \
                                                    (num_obs, num_sequences, horizon, self.ac_dim))
            mean = np.mean(candidate_sequences, axis=0)
            std = np.std(candidate_sequences, axis=0)
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                rewards = self.evaluate_candidate_sequences(candidate_sequences, obs)
                elite_idx = np.argsort(rewards)[-self.cem_num_elites:]
                elite_sequences = candidate_sequences[elite_idx]
                if i == self.cem_iterations - 1:
                    break
                mean *= 1 - self.cem_alpha
                mean += self.cem_alpha * np.mean(elite_sequences, axis=0)
                std *= 1 - self.cem_alpha
                std *= self.cem_alpha * np.std(elite_sequences, axis=0)
                candidate_sequences = np.random.normal(mean, std, (num_sequences, horizon, self.ac_dim))
                candidate_sequences = np.clip(candidate_sequences, self.low, self.high)

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            cem_action = elite_sequences[0]
            return cem_action[None]

        elif self.sample_strategy == 'mppi':
            # Shape of mppi_mean: (num_obs, horizon, ac_dim)
            past_action = self.mppi_mean[:, 0, :].copy() # Shape: (obs_num, ac_dim)
            self.mppi_mean[:, :-1, :] = self.mppi_mean[:, 1:, :]
            # Sample noise
            if(self.sample_velocity):
                mu_higherRange = np.random.normal(0, self.mppi_sigma, \
                                                  size=(num_obs, num_sequences, self.horizon, self.ac_dim))
                lowerRange = 0.3 * self.mppi_sigma
                num_lowerRange = int(0.1 * num_sequences)
                mu_lowerRange = np.random.normal(0, lowerRange, (num_obs, num_lowerRange, self.horizon, self.ac_dim))
                mu_higherRange[:, -num_lowerRange:, ...] = mu_lowerRange
                mu = mu_higherRange.copy()
            else:
                mu = np.random.normal(0, self.mppi_sigma, (num_obs, num_sequences, self.horizon, self.ac_dim))

            # sample candidate sequences
            candidate_sequences = mu.copy()
            for i in range(self.horizon):
                if i == 0:
                    candidate_sequences[..., i, :] = self.mppi_beta * (self.mppi_mean[:, None, i, :] + mu[..., i, :])
                    candidate_sequences[..., i, :] += (1 - self.mppi_beta) * past_action[:, None, :]
                else:
                    candidate_sequences[..., i, :] = self.mppi_beta * (self.mppi_mean[:, None, i, :] + mu[..., i, :]) 
                    candidate_sequences[..., i, :] += (1 - self.mppi_beta) * candidate_sequences[..., i - 1, :]
            # shape of candidate_sequences: (num_obs, num_sequences, horizon, ac_dim)
            candidate_sequences = np.clip(candidate_sequences, self.low, self.high) 
            
            # calculate the reward of every sequences
            rewards = self.evaluate_candidate_sequences(candidate_sequences, obs) # Shape: (num_obs, num_sequences)

            # update path
            max_reward = np.max(rewards, axis=-1)[:, None]
            S = np.exp(self.mppi_gamma * (rewards - max_reward)) # Shape: (num_obs, num_sequences)
            partition = np.sum(S, axis=1)[:, None] + 1e-10
            # Shape of weighted_actions: (num_obs, num_sequences, H, ac_dim)
            weighted_actions = (candidate_sequences * (S / partition)[:, :, None, None]) 
            self.mppi_mean = np.sum(weighted_actions, axis=1)  # Shape: (num_obs, H, ac_dim)

            return self.mppi_mean.copy()[:, None, ...] # Shape: (num_obs, 1, horizon, ac_dim)

        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (num_obs, num_sequences)

        if self.model is not None:
            sum_of_rewards = self.model_forward(obs, candidate_action_sequences)
        elif self.evaluator is not None:
            sum_of_rewards = self.evaluator.run(obs, candidate_action_sequences)
        else:
            raise NotImplementedError
        return sum_of_rewards

    def model_forward(self, obs, ac_seq: np.ndarray):
        """
        Input of a forward function:
            The current observation(s) of shape: (M, xxx):
                - M: the number of states
            Several action sequences of shape (M, N, H, ac_dim):
                - N: the number of action / action sequences
                - H: the lenth of an action sequence
                - ac_dim: the dimensionality of the action space
        Outupt of a forward function:
            The summation of rewards of each action sequences of shape (N,)
        """
        assert len(ac_seq.shape) == 4, "Action sequences need to be of shape (M, N, H, ac_dim)!"
        obs = stack([obs] * ac_seq.shape[1], axis=1) # Shape: (M, N, xxx)

        if isinstance(self.model, PointNetModel) or isinstance(self.model, CustomDataParallel):
            if hasattr(self.model, "update_norm"):
                ds_std = self.model.ds_std
                ds_mean = self.model.ds_mean
                reward_std = self.model.reward_std
                reward_mean = self.model.reward_mean
                add_ori = self.model.add_ori
            else:
                ds_std = self.model.module.ds_std
                ds_mean = self.model.module.ds_mean
                reward_std = self.model.module.reward_std
                reward_mean = self.model.module.reward_mean
                add_ori = self.model.module.add_ori

            obs = to_torch(obs, dtype='float32', device=self.device, non_blocking=True)
            if add_ori: ids = obs.pop("id") # Shape of ids: (M, N, 2)
            ac_seq = to_torch(ac_seq, dtype='float32', device=self.device, non_blocking=True)

            with torch.no_grad():
                total_reward = 0
                for i in range(self.num_ensemble):
                    obs_this_ensemble = copy.deepcopy(obs)
                    reward_this_ensemble = 0
                    for t in range(ac_seq.shape[2]):
                        output = self.model(obs_this_ensemble, ac_seq[..., t, :], ids, i)
                        obs_this_ensemble['pointcloud']['xyz'] += output['dp']
                        obs_this_ensemble['state'] += output['ds'] * ds_std + ds_mean
                        reward_this_ensemble += output['rewards'] * reward_std + reward_mean
                    total_reward += reward_this_ensemble
            return to_np(total_reward / self.num_ensemble, dtype='float32')

        elif isinstance(self.model, list): # model is a list of workers
            assert len(self.model) == self.num_ensemble
            for i in range(self.num_ensemble):
                self.model[i].call('forward', obs, ac_seq)
            total_reward = 0
            for i in range(self.num_ensemble):
                total_reward += self.model[i].get() # np.ndarray
            return total_reward / self.num_ensemble

        else:
            raise NotImplementedError

    def update_norm(self, reward_mean, reward_std, ds_mean, ds_std):
        if hasattr(self.model, "update_norm"):
            self.model.update_norm(reward_mean, reward_std, ds_mean, ds_std)
        else:
            self.model.module.update_norm(reward_mean, reward_std, ds_mean, ds_std)