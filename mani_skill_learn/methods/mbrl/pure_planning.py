from ..builder import MBRL
from ...env.env_utils import build_env, true_done
from ...networks.policy_network import MPCPolicy
import numpy as np
import time, math, copy


@MBRL.register_module()
class PurePlanning():
    def __init__(self, policy_cfg, env_cfg, obs_shape, action_shape, action_space, num_procs=1):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_space = action_space

        # Parallel or not?
        if num_procs > 1:
            self.evaluator = BatchEvaluator(env_cfg, num_procs)
        else:
            self.evaluator = Evaluator(env_cfg)
        
        self.policy = MPCPolicy(ac_dim=action_shape, ac_space=action_space,
                                evaluator=self.evaluator, **policy_cfg)
        self.policy.reset()

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, obs, num_actions=1, mode=None):
        return self.policy(obs, num_actions, mode)



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
        sequences. Shape [B, H, D_action] or [M, B, H, D_action] where
            - M = 1
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
        M = 0
        if len(init_obs.shape) == 2 or len(candidate_action_sequences.shape) == 4:
            assert len(init_obs.shape) == 2 and len(candidate_action_sequences.shape) == 4
            assert init_obs.shape[0] == candidate_action_sequences.shape[0] == 1
            init_obs = init_obs[0]
            M, N, H, ac_dim = candidate_action_sequences.shape
            candidate_action_sequences = candidate_action_sequences.reshape(N, H, ac_dim)

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

        return sum_of_rewards.reshape(M, N) if M > 0 else sum_of_rewards

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
        M = 0
        if len(init_obs.shape) == 2 or len(candidate_action_sequences.shape) == 4:
            assert len(init_obs.shape) == 2 and len(candidate_action_sequences.shape) == 4
            assert init_obs.shape[0] == candidate_action_sequences.shape[0] == 1
            init_obs = init_obs[0]
            M, N, H, ac_dim = candidate_action_sequences.shape
            candidate_action_sequences = candidate_action_sequences.reshape(N, H, ac_dim)

        N = candidate_action_sequences.shape[0]
        batchsize = math.ceil(N / self.num_procs)
        sum_of_rewards = []
        for i in range(self.num_procs):
            batch_sequences = candidate_action_sequences[i * batchsize: min((i + 1) * batchsize, N)]
            self.workers[i].call('run', init_obs, batch_sequences)
        for i in range(self.num_procs):
            sum_of_rewards.append(self.workers[i].get())
        
        if M > 0:
            return np.concatenate(sum_of_rewards, axis=0).reshape(M, N)
        return np.concatenate(sum_of_rewards, axis=0)