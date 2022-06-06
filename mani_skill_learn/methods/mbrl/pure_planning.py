from ..builder import MBRL
from ...env.env_utils import build_env, true_done
import numpy as np
import time, math, copy


@MBRL.register_module()
class PurePlanning():
    def __init__(self, policy_cfg, env_cfg, obs_shape, action_shape, action_space):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_space = action_space
        self.policy = MPCPolicy(ac_dim=action_shape, ac_space=action_space, env_cfg=env_cfg, **policy_cfg)

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, obs, num_actions=1, mode=None):
        return self.policy(obs)

# Adapted from the homework 4 of cs285 UC Berkeley (2021 fall), which is completed by 
# Deming Chen (Email: cdm@pku.edu.cn). Original code of the homework:
# https://github.com/berkeleydeeprlcourse/homework_fall2021/tree/main/hw4/cs285/models
class MPCPolicy():
    def __init__(self,                 
                 ac_dim,
                 ac_space,
                 horizon,
                 num_action_sequences,
                 cem_cfg,
                 mppi_cfg,
                 env_cfg,
                 num_procs=1,
                 env=None,
                 sample_strategy='random'
                 ):

        # init vars
        self.env = env
        self.horizon = horizon
        self.N = num_action_sequences
        self.num_procs = num_procs

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
            self.mppi_sigma = mppi_cfg['mag_noise'] * (self.high - self.low) / 2
            self.mppi_mean = np.tile((self.high + self.low) / 2, (self.horizon, 1))

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")
        if self.sample_strategy == 'mppi':
            print(f"MPPI params: gamma={self.mppi_gamma}, beta={self.mppi_beta}, sigma={self.mppi_sigma}")

        # Parallel or not?
        if num_procs > 1:
            self.evaluator = BatchEvaluator(env_cfg, num_procs)
        else:
            self.evaluator = Evaluator(env_cfg)

    #get_action
    def __call__(self, obs, num_actions=1, mode=None):
        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs.flatten())

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)] # TODO (Q2)
            action_to_take = best_action_sequence[0]  # TODO (Q2)
            return action_to_take[None]  # Unsqueeze the first index

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or ((self.sample_strategy == 'cem' or self.sample_strategy == 'mppi') 
            and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
            return random_action_sequences

        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            candidate_sequences = np.random.uniform(self.low, self.high, (num_sequences, horizon, self.ac_dim))
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
            past_action = self.mppi_mean[0].copy()
            self.mppi_mean[:-1] = self.mppi_mean[1:]
            # Sample noise
            if(self.sample_velocity):
                mu_higherRange = np.random.normal(0, self.mppi_sigma, size=(num_sequences, self.horizon, self.ac_dim))
                lowerRange = 0.3 * self.mppi_sigma
                num_lowerRange = int(0.1 * num_sequences)
                mu_lowerRange = np.random.normal(0, lowerRange, size=(num_lowerRange, self.horizon, self.ac_dim))
                mu_higherRange[-num_lowerRange:] = mu_lowerRange
                mu = mu_higherRange.copy()
            else:
                mu = np.random.normal(0, self.sigma, size=(self.N, self.horizon, self.ac_dim))
            
            # sample candidate sequences
            candidate_sequences = mu.copy()
            for i in range(self.horizon):
                if i == 0:
                    candidate_sequences[:, i] = self.mppi_beta*(self.mppi_mean[i] + mu[:, i])
                    candidate_sequences[:, i] += (1 - self.mppi_beta) * past_action
                else:
                    candidate_sequences[:, i] = self.mppi_beta*(self.mppi_mean[i] + mu[:, i]) 
                    candidate_sequences[:, i] += (1 - self.mppi_beta) * candidate_sequences[:, i - 1]
            candidate_sequences = np.clip(candidate_sequences, self.low, self.high) # shape: (N, horizon, ac_dim)
            
            # calculate the reward of every sequences
            rewards = self.evaluate_candidate_sequences(candidate_sequences, obs) # Shape: (N,)

            # update path
            S = np.exp(self.mppi_gamma * (rewards - np.max(rewards))) # Shape: (N,)
            partition = np.sum(S) + 1e-10
            weighted_actions = (candidate_sequences * (S / partition)[:, None, None]) # Shape: (N, H, ac_dim)
            self.mppi_mean = np.sum(weighted_actions, axis=0)  # Shape: (H, ac_dim)

            return self.mppi_mean.copy()[None]

        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        #
        # NOTE: In this algorithm, we assume no model, and use the environment to get dynamics and reward 
        # directly. So the code below is adapted by Deming Chen

        reward = self.calculate_sum_of_rewards(obs, candidate_action_sequences)
        return reward

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences):
        parallel_rewards = self.evaluator.run(obs, candidate_action_sequences)
        return parallel_rewards

class Evaluator():
    def __init__(self, env_cfg, worker_id=None) -> None:
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg['unwrapped'] = False
        self.env = build_env(env_cfg)
        self.env.reset()
        self.env_name = env_cfg.env_name
        self.worker_id = worker_id

    def run(self, init_obs, candidate_action_sequences):
        """
        :param init_obs: numpy array with the initial observation of a sequence. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [B, H, D_action] where
            - B is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [B].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        #
        sum_of_rewards = np.zeros(candidate_action_sequences.shape[0])
        for i in range(candidate_action_sequences.shape[0]):
            self.env.set_state(init_obs)
            for t in range(candidate_action_sequences.shape[1]): # for t = 0, 1, ..., Horizen - 1
                ob, r, done, info = self.env.step(candidate_action_sequences[i, t])
                sum_of_rewards[i] += r
                if true_done(done, info):
                    break
            self.env.change_step_in_ep(0)
        self.env.reset()
        return sum_of_rewards

class BatchEvaluator():
    def __init__(self, env_cfg, num_procs=1):
        self.env_name = env_cfg.env_name
        self.num_procs = num_procs

        from ...env.parallel_runner import NormalWorker as Worker
        self.workers = []
        for i in range(num_procs):
            self.workers.append(Worker(Evaluator, i, env_cfg=env_cfg))

    def run(self, init_obs, candidate_action_sequences):
        """
        :param init_obs: numpy array with the initial observation of a sequence. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N = candidate_action_sequences.shape[0]
        batchsize = math.ceil(N / self.num_procs)
        sum_of_rewards = []
        for i in range(self.num_procs):
            batch_sequences = candidate_action_sequences[i * batchsize: min((i + 1) * batchsize, N)]
            self.workers[i].call('run', init_obs, batch_sequences)
        for i in range(self.num_procs):
            sum_of_rewards.append(self.workers[i].get())
        
        return np.concatenate(sum_of_rewards, axis=0)