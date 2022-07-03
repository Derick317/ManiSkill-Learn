import time, random, copy, numpy as np
from tqdm.auto import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import MBRL
from mani_skill_learn.networks import build_model, hard_update, soft_update
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch, compare_dict_array, add_dict_array
from mani_skill_learn.utils.data import sample_element_in_dict_array
from mani_skill_learn.utils.data.concat import concat_list_of_array as concat
from mani_skill_learn.utils.data.concat import stack_list_of_array as stack
from mani_skill_learn.utils.math import split_num
from ...networks.policy_network import MPCPolicy, RandPolicy
from mani_skill_learn.utils.torch import BaseAgent, save_checkpoint, load_checkpoint
from mani_skill_learn.env.torch_parallel_runner import TorchWorker as Worker
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

@MBRL.register_module()
class REPLAN(BaseAgent):
    def __init__(self, 
                 policy_cfg, 
                 rand_policy_cfg, 
                 model_cfg, 
                 obs_shape, 
                 action_shape, 
                 action_space,
                 batchsize, 
                 loss_coeff: dict,
                 num_ensemble=1,
                 add_ori=False,
                 del_rgb=False, 
                 nEpoch=None, 
                 nstep=None):
        super(REPLAN, self).__init__()
        self.nEpoch = nEpoch # How many epoches when training the model?
        self.nstep = nstep # How many times is the model updated when training the model?
        self.batchsize = batchsize
        self.loss_coeff = loss_coeff
        self.del_rgb = del_rgb
        self.add_ori = add_ori
        self.num_ensemble = num_ensemble
        if nEpoch is not None: self.epoch_counter = 0

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_space = action_space
        model_optim_cfg = model_cfg.pop("optim_cfg")

        model_cfg['obs_shape'] = obs_shape
        model_cfg['action_shape'] = action_shape
        model_cfg['add_ori'] = add_ori
        model_cfg['num_ensemble'] = num_ensemble
        model_cfg['del_rgb'] = del_rgb
        policy_cfg['num_ensemble'] = num_ensemble
        self.model = build_model(model_cfg)
        self.policy = MPCPolicy(ac_dim=action_shape, ac_space=action_space, model=self.model, **policy_cfg)
        self.rand_policy = RandPolicy(action_space, rand_policy_cfg)
        self.model_optim = build_optimizer(self.model, model_optim_cfg)

    def train_model(self, buffers):
        trainOnPol = buffers["trainOnPol"]
        valOnPol = buffers["valOnPol"]
        trainRand = buffers["trainRand"]
        valRand = buffers["valRand"]
        
        if isinstance(self.nEpoch, int) and self.nstep is None:
            train_loss, valOnPol_loss, valRand_loss = [], [], []
            for _ in trange(self.nEpoch, desc="Train model", leave=False):
                self.epoch_counter += 1

                # Train the model
                batches, re_mean, re_std, ds_mean, ds_std = data_process([trainRand, trainOnPol], self.batchsize)
                self.update_norm(re_mean, re_std, ds_mean, ds_std)
                idx = list(range(self.num_ensemble))
                random.shuffle(idx)
                epoch_train_loss = {}
                for i in range(len(batches)):
                    # tmp_time = time.time()
                    data = batches[i]
                    data = to_torch(data, dtype='float32', device=self.device, non_blocking=True)
                    if self.add_ori:
                        if "id" in data["obs"]:
                            ids = data["obs"].pop("id")
                        else: raise "Not found the ids of the cabinets or the doors!" 
                    
                    for key in data:
                        if not isinstance(data[key], dict) and data[key].ndim == 1:
                            data[key] = data[key][..., None]
                    
                    outputs = self.model(data["obs"], data["actions"], ids, idx[i % self.num_ensemble])

                    # Calculate loss and back propagate
                    ground_truth = get_ground_truth(data)
                    batch_loss = calculate_loss(self.loss_coeff, outputs, ground_truth)
                    self.model_optim.zero_grad()
                    batch_loss['total'].backward()
                    self.model_optim.step()
                    epoch_train_loss = add_dict_array(epoch_train_loss, batch_loss)

                    # batch_time = time.time() - tmp_time
                    # print("Time for one batch: ", batch_time)

                for each_loss in epoch_train_loss.values(): each_loss /= len(batches)
                train_loss.append((self.epoch_counter, epoch_train_loss))

                # Validate the model
                # Validation loss on random set
                num_batch = min(2000, len(valRand))
                valRand_batches = data_process([valRand], num_batch, num_batch)[0][0]
                valRand_data = to_torch(valRand_batches, dtype='float32', device=self.device, non_blocking=True)
                if self.add_ori: ids = valRand_data["obs"].pop("id")
                ground_truth = get_ground_truth(valRand_data)
                with torch.no_grad():
                    valRand_outputs = concat(self.model(valRand_data["obs"], valRand_data["actions"], ids), axis=0)
                    epoch_valRand_loss = calculate_loss(self.loss_coeff, valRand_outputs, ground_truth)
                valRand_loss.append((self.epoch_counter, epoch_valRand_loss))

                # Validation loss on on-policy set
                num_batch = min(2000, len(valOnPol))
                valOnPol_batches = data_process([valOnPol], num_batch, num_batch)[0][0]
                valOnPol_data = to_torch(valOnPol_batches, dtype='float32', device=self.device, non_blocking=True)
                if self.add_ori: ids = valOnPol_data["obs"].pop("id")
                ground_truth = get_ground_truth(valOnPol_data)
                with torch.no_grad():
                    valOnPol_outputs = concat(self.model(valOnPol_data["obs"], valOnPol_data["actions"], ids), axis=0)
                    epoch_valOnPol_loss = calculate_loss(self.loss_coeff, valOnPol_outputs, ground_truth)
                valOnPol_loss.append((self.epoch_counter, epoch_valOnPol_loss))
        elif isinstance(self.nstep, int) and self.nEpoch is None:
            raise NotImplementedError
        else: 
            raise TypeError("Cannot determine how many times the model is updated when training.") 

        return train_loss, valOnPol_loss, valRand_loss

    def update_norm(self, reward_mean, reward_std, ds_mean, ds_std):
        self.policy.update_norm(reward_mean, reward_std, ds_mean, ds_std)

    def reset(self, *args, **reset_kwargs):
        self.policy.reset(*args, **reset_kwargs)
    
@MBRL.register_module()
class REPLANV2():
    """
    We want distributed training, so let's load each model in ensemble to different GPU.
    """
    def __init__(self, policy_cfg, rand_policy_cfg, model_cfg, 
                 obs_shape, action_shape, action_space,
                 batchsize, loss_coeff: dict, num_ensemble=1, add_ori=False, del_rgb=False, nEpoch=1):
        self.batchsize = batchsize
        self.del_rgb = del_rgb
        self.num_ensemble = num_ensemble
        self.epoch_counter = 0
        self.nEpoch = nEpoch

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_space = action_space
        model_optim_cfg = model_cfg.pop("optim_cfg")

        model_cfg['obs_shape'] = obs_shape
        model_cfg['action_shape'] = action_shape
        model_cfg['add_ori'] = add_ori
        model_cfg['num_ensemble'] = 1
        model_cfg['del_rgb'] = del_rgb
        policy_cfg['num_ensemble'] = num_ensemble

        self.models = []
        for i in range(num_ensemble):
            self.models.append(Worker(ReplanV2Model, i, nEpoch, add_ori, model_cfg, model_optim_cfg, loss_coeff))
        self.policy = MPCPolicy(ac_dim=action_shape, ac_space=action_space, model=self.models, **policy_cfg)
        self.rand_policy = RandPolicy(action_space, rand_policy_cfg)

    def train_model(self, buffers):
        trainOnPol = buffers["trainOnPol"]
        valOnPol = buffers["valOnPol"]
        trainRand = buffers["trainRand"]
        valRand = buffers["valRand"]

        # Get and distribute data from buffers
        train_batches, re_mean, re_std, ds_mean, ds_std = data_process([trainRand, trainOnPol], self.batchsize)
        statistics = (re_mean, re_std, ds_mean, ds_std)
        _, num_batch = split_num(len(train_batches), self.num_ensemble)
        ensemble_train_batches, start = [], 0
        for i in range(self.num_ensemble):
            ensemble_train_batches.append(train_batches[start: start + num_batch[i]])
            start += num_batch[i]
        random.shuffle(ensemble_train_batches)
        valOnPol_batches = data_process([valOnPol], min(1000, len(valOnPol)), min(2000, len(valOnPol)))[0]
        valRand_batches = data_process([valOnPol], min(1000, len(valRand)), min(2000, len(valRand)))[0]

        # Start training
        for i in range(self.num_ensemble):
            self.models[i].call('train_model', statistics, ensemble_train_batches[i], valOnPol_batches, valRand_batches)
        ensemble_train_loss, ensemble_valOnPol_loss, ensemble_valRand_loss = [], [], []
        for i in range(self.num_ensemble):
            train_loss, valOnPol_loss, valRand_loss = self.models[i].get()
            ensemble_train_loss.append(train_loss)
            ensemble_valOnPol_loss.append(valOnPol_loss)
            ensemble_valRand_loss.append(valRand_loss)

        # Merge loss
        start_epoch = ensemble_valRand_loss[0][0][0]
        end_epoch = ensemble_valRand_loss[0][-1][0]
        train_loss, valOnPol_loss, valRand_loss = [], [], []
        for j in range(end_epoch - start_epoch + 1):
            train_loss_sum, valOnPol_loss_sum, valRand_loss_sum = dict(), dict(), dict()
            for i in range(self.num_ensemble):
                assert ensemble_train_loss[i][j][0] == j + start_epoch
                assert ensemble_valOnPol_loss[i][j][0] == j + start_epoch
                assert ensemble_valRand_loss[i][j][0] == j + start_epoch
                train_loss_sum = add_dict_array(ensemble_train_loss[i][j][1], train_loss_sum)
                valOnPol_loss_sum = add_dict_array(ensemble_valOnPol_loss[i][j][1], valOnPol_loss_sum)
                valRand_loss_sum = add_dict_array(ensemble_valRand_loss[i][j][1], valRand_loss_sum)
            for key in train_loss_sum:
                train_loss_sum[key] /= self.num_ensemble
                valOnPol_loss_sum[key] /= self.num_ensemble
                valRand_loss_sum[key] /= self.num_ensemble
            train_loss.append((j + start_epoch, train_loss_sum))
            valOnPol_loss.append((j + start_epoch, valOnPol_loss_sum))
            valRand_loss.append((j + start_epoch, valRand_loss_sum))
                
        return train_loss, valOnPol_loss, valRand_loss 

    def __call__(self, obs, num_actions=1, mode=None):
        return self.policy(obs, num_actions, mode)

    def reset(self, *args, **reset_kwargs):
        self.policy.reset(*args, **reset_kwargs)

    def save_checkpoint(self, model_path):
        assert model_path[-5:] == '.ckpt'
        for i in range(self.num_ensemble):
            self.models[i].call('save_checkpoint', model_path)
        for i in range(self.num_ensemble):
            self.models[i].get()

    def load_checkpoint(self, filename):
        assert filename[-5:] == ".ckpt"
        for i in range(self.num_ensemble):
            self.models[i].call('load_checkpoint', filename)
        for i in range(self.num_ensemble):
            self.models[i].get()

    def to(self, device: str):
        if device == 'cuda':
            for i in range(self.num_ensemble):
                self.models[i].call('to_cuda')
        elif device == 'cpu':
            for i in range(self.num_ensemble):
                self.models[i].call('to_cpu')
        else:
            raise NotImplementedError
        for i in range(self.num_ensemble):
            self.models[i].get()

    def change_nEpoch(self, nEpoch):
        self.nEpoch = nEpoch
        for i in range(self.num_ensemble):
            self.models[i].call('change_nEpoch', nEpoch)
        for i in range(self.num_ensemble):
            self.models[i].get()

    def eval(self):
        for i in range(self.num_ensemble):
            self.models[i].call('eval')
        for i in range(self.num_ensemble):
            self.models[i].get()
    
    def train(self):
        for i in range(self.num_ensemble):
            self.models[i].call('train')
        for i in range(self.num_ensemble):
            self.models[i].get()

class ReplanV2Model():
    def __init__(self, nEpoch, add_ori, 
                 model_cfg, model_optim_cfg, loss_coeff: dict, worker_id=None):
        self.worker_id = worker_id
        self.model = build_model(model_cfg)
        self.model_optim = build_optimizer(self.model, model_optim_cfg)
        self.loss_coeff = loss_coeff
        self.nEpoch = nEpoch
        self.add_ori = add_ori
        self.epoch_counter = 0

    def forward(self, obs: np.ndarray, ac_seq: np.ndarray):
        """
        Input of a forward function:
            The current observation(s) of shape: (M, N, xxx):
                - M: the number of states
                - N: the number of action / action sequences
            Several action sequences of shape (M, N, H, ac_dim):
                - H: the lenth of an action sequence
                - ac_dim: the dimensionality of the action space
        Outupt of a forward function:
            The summation of rewards of each action sequences of shape (M, N)
        """
        obs = to_torch(obs, dtype='float32', device=self.device, non_blocking=True)
        if self.model.add_ori: ids = obs.pop("id") # Shape of ids: (M, N, 2)
        ac_seq = to_torch(ac_seq, dtype='float32', device=self.device, non_blocking=True)
        obs_this_ensemble = copy.deepcopy(obs)
        reward_this_ensemble = 0
        with torch.no_grad():
            for t in range(ac_seq.shape[2]):
                output = self.model(obs_this_ensemble, ac_seq[..., t, :], ids, 0)
                obs_this_ensemble['pointcloud']['xyz'] += output['dp']
                obs_this_ensemble['state'] += output['ds'] * self.model.ds_std + self.model.ds_mean
                reward_this_ensemble += output['rewards'] * self.model.reward_std + self.model.reward_mean
        return reward_this_ensemble.detach().cpu().numpy()

    def train_model(self, statistics, train_batches: list, valOnPol_batches: list, valRand_batches: list):
        """
        Input: 
            - statistics: tuple (re_mean, re_std, ds_mean, ds_std)
            - xxx_batches: lists of dict of arrays, and each array is a numpy ndarray.
        output: dictionaries of key: 'total', 'rewards', 'ds', and 'next_pcd'
        """
        train_loss, valOnPol_loss, valRand_loss = [], [], []
        re_mean, re_std, ds_mean, ds_std = statistics
        self.update_norm(re_mean, re_std, ds_mean, ds_std)
        for _ in range(self.nEpoch):
            self.epoch_counter += 1
            epoch_train_loss, epoch_valOnPol_loss, epoch_valRand_loss = dict(), dict(), dict()

            # Train the model
            
            for i in range(len(train_batches)):
                data = train_batches[i]
                data = to_torch(data, dtype='float32', device=self.model.device, non_blocking=True)
                if self.add_ori:
                    if "id" in data["obs"]:
                        ids = data["obs"].pop("id")
                    else: raise "Not found the ids of the cabinets or the doors!" 
                
                for key in data:
                    if not isinstance(data[key], dict) and data[key].ndim == 1:
                        data[key] = data[key][..., None]
                
                outputs = self.model(data["obs"], data["actions"], ids, 0)

                # Calculate loss and back propagate
                ground_truth = get_ground_truth(data)
                batch_loss = calculate_loss(self.loss_coeff, outputs, ground_truth)
                self.model_optim.zero_grad()
                batch_loss['total'].backward()
                self.model_optim.step()
                for key in batch_loss: batch_loss[key] = float(batch_loss[key].detach())
                epoch_train_loss = add_dict_array(epoch_train_loss, batch_loss)

            for key in epoch_train_loss: epoch_train_loss[key] /= len(train_batches) 
            train_loss.append((self.epoch_counter, epoch_train_loss))

            # Validate the model
            # Validation loss on random set
            for valRand_batch in valRand_batches:
                valRand_data = to_torch(valRand_batch, dtype='float32', device=self.model.device, non_blocking=True)
                if self.add_ori: ids = valRand_data["obs"].pop("id")
                ground_truth = get_ground_truth(valRand_data)
                with torch.no_grad():
                    valRand_outputs = concat(self.model(valRand_data["obs"], valRand_data["actions"], ids), axis=0)
                    batch_valRand_loss = calculate_loss(self.loss_coeff, valRand_outputs, ground_truth)
                for key in batch_valRand_loss:
                    batch_valRand_loss[key] = float(batch_valRand_loss[key].detach()) / len(valRand_batches)
                epoch_valRand_loss = add_dict_array(epoch_valRand_loss, batch_valRand_loss)
            valRand_loss.append((self.epoch_counter, epoch_valRand_loss))

            # Validation loss on on-policy set
            for valOnPol_batch in valOnPol_batches:
                valOnPol_data = to_torch(valOnPol_batch, dtype='float32', device=self.model.device, non_blocking=True)
                if self.add_ori: ids = valOnPol_data["obs"].pop("id")
                ground_truth = get_ground_truth(valOnPol_data)
                with torch.no_grad():
                    valOnPol_outputs = concat(self.model(valOnPol_data["obs"], valOnPol_data["actions"], ids), axis=0)
                    batch_valOnPol_loss = calculate_loss(self.loss_coeff, valOnPol_outputs, ground_truth)
                for key in batch_valOnPol_loss:
                    batch_valOnPol_loss[key] = float(batch_valOnPol_loss[key].detach()) / len(valOnPol_batches)
                epoch_valOnPol_loss = add_dict_array(epoch_valOnPol_loss, batch_valOnPol_loss)
            valOnPol_loss.append((self.epoch_counter, epoch_valOnPol_loss))
        
        return train_loss, valOnPol_loss, valRand_loss

    def update_norm(self, reward_mean, reward_std, ds_mean, ds_std):
        self.model.update_norm(reward_mean, reward_std, ds_mean, ds_std)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def change_nEpoch(self, nEpoch):
        self.nEpoch = nEpoch

    def to_cuda(self):
        self.model.to(torch.device('cuda:' + str(self.worker_id)))

    def to_cpu(self):
        self.model.to(torch.device('cpu'))

    def save_checkpoint(self, model_path):
        model_path = model_path[:-5] + f'({self.worker_id})' + '.ckpt'
        save_checkpoint(self.model, model_path)

    def load_checkpoint(self, filename):
        filename = filename[:-5] + f'({self.worker_id})' + '.ckpt'
        load_checkpoint(self.model, filename, map_location='cpu')

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

def calculate_loss(loss_coeff, pred, ground_truth):
    ChamLoss = chamfer_3DDist()
    loss_dr = F.mse_loss(ground_truth['rewards'].squeeze(), pred['rewards'])
    loss_ds = F.mse_loss(ground_truth['ds'], pred['ds'])
    dist1, dist2, _, _ = ChamLoss(ground_truth['next_pcd'], pred['dp'] + ground_truth['pointcloud'])
    chamfer = torch.mean(dist1) + torch.mean(dist2)
    total_loss = loss_dr * loss_coeff['rewards'] + loss_ds * loss_coeff['ds']
    total_loss += chamfer * loss_coeff['chamfer']
    loss_dict = {"total": total_loss, "rewards": loss_dr, "ds": loss_ds, "next_pcd": chamfer}

    return loss_dict

def get_ground_truth(data, start=0, end=None):
    ground_truth = dict()
    if end is not None:
        ground_truth['rewards'] = data['rewards'][start: end]
        ground_truth['ds'] = data['ds'][start: end]
        ground_truth['pointcloud'] = data['obs']['pointcloud']['xyz'][start: end]
        ground_truth['next_pcd'] = data['next_pcd']['xyz'][start: end]
    else:
        ground_truth['rewards'] = data['rewards'][start:]
        ground_truth['ds'] = data['ds'][start:]
        ground_truth['pointcloud'] = data['obs']['pointcloud']['xyz'][start:]
        ground_truth['next_pcd'] = data['next_pcd']['xyz'][start:]
    
    return ground_truth

def data_process(buffers: list, batchsize, total_num=None, rand=True):
    """
    Process data from rollouts to training set or validation set.
    Specifically, we first concatenate the data in all buffers, then calculate the mean
    and the standard deviation of the statistics, and get their normalized version and
    cut them into several batches.

    Rollouts in buffers have the following structure:
        - obs:
            - state
            - pointcloud:
                - xyz
                - rgb (if not be deleted)
                - seg
            - id
        - actions
        - rewards
        - next_obs:
            - state
            - pointcloud:
                - xyz
                - rgb (if not be deleted)
                - seg
            - id
    
    The output data is a list containing all batches, each of which has the following
    structure:
        - obs:
            - state
            - pointcloud
            - id
        - actions
        - rewards: normalized rewards
        - ds: normalized difference of the next state and the current state
        - next_pcd
    """
    buffer_content = []
    if total_num == None:
        for buffer in buffers: 
            if len(buffer) > 0: buffer_content.append(buffer.get_all())
        assert len(buffer_content) > 0
    else:
        for buffer in buffers: 
            if len(buffer) > 0: buffer_content.append(buffer.sample(total_num))
        assert len(buffer_content) == 1

    rollout = concat(buffer_content, axis=0)
    idx = list(range(rollout['actions'].shape[0]))
    if rand: random.shuffle(idx)

    obs = rollout['obs']
    actions = rollout['actions']
    reward_unnorm = rollout['rewards']
    ds_unnorm = rollout['next_obs']['state'] - rollout['obs']['state']
    next_pcd = rollout['next_obs']['pointcloud']
    ds_mean = np.mean(ds_unnorm, axis=0)
    ds_std = np.std(ds_unnorm, axis=0)
    re_mean = np.mean(reward_unnorm, axis=0)
    re_std = np.std(reward_unnorm, axis=0)
    rewards = (reward_unnorm - re_mean) / (re_std + 1e-8)
    ds = (ds_unnorm - ds_mean) / (ds_std + 1e-8)
    data = {'obs': obs, 'actions': actions, 'rewards': rewards, 'ds': ds, 'next_pcd': next_pcd}

    num_batch = data['rewards'].shape[0] // batchsize
    _, len_batch = split_num(len(idx), num_batch)
    samples = []
    start = 0
    for i in range(num_batch):
        samples.append(sample_element_in_dict_array(data, idx[start: start + len_batch[i]]))
        start += len_batch[i]
    
    return samples, re_mean, re_std, ds_mean, ds_std