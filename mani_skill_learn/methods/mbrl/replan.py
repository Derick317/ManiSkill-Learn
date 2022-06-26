import time, random, copy, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import MBRL
from mani_skill_learn.networks import build_model, hard_update, soft_update
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch, compare_dict_array, add_dict_array, split_in_dict_array
from mani_skill_learn.utils.data.concat import concat_list_of_array as concat
from mani_skill_learn.utils.data.concat import stack_list_of_array as stack
from ...networks.policy_network import MPCPolicy, RandPolicy
from mani_skill_learn.utils.torch import BaseAgent
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
            for _ in range(self.nEpoch):
                self.epoch_counter += 1

                # Train the model
                B_Onpol = len(trainOnPol) * self.batchsize // (len(trainRand) + len(trainOnPol))
                B_Rand = (len(trainRand) * self.batchsize - 1) // (len(trainRand) + len(trainOnPol)) + 1
                trainOnPol_batches = trainOnPol.get_all_batch(B_Onpol)
                trainRand_batches = trainRand.get_all_batch(B_Rand)
                # trainOnPol_batches_copy = copy.deepcopy(trainOnPol_batches)
                # trainRand_batches_copy = copy.deepcopy(trainRand_batches)
                assert len(trainOnPol_batches) >= len(trainRand_batches)
                num_batch = len(trainOnPol_batches)
                epoch_train_loss = {}
                loss_dict = dict()
                for key in self.loss_coeff: loss_dict[key] = 0
                for i in range(num_batch):
                    # concatenate on-policy set and random set together
                    if i < len(trainRand_batches):
                        data = concat([trainOnPol_batches[i], trainRand_batches[i]], axis=0)
                    else:
                        data = trainOnPol_batches[i]
                    len_data = data['rewards'].shape[0]
                    
                    # Because of some parallel tricks, we have to train each model in the
                    # ensemble seperately, otherwise we do not know the batchsize of each 
                    # model.
                    # Let's split the data into self.num_ensemble pieces at first.
                    data_ensemble = split_in_dict_array(data, (len_data - 1) // self.num_ensemble + 1)
                    assert len(data_ensemble) == self.num_ensemble
                    random.shuffle(data_ensemble)
                    # data_ensemble is a list
                    batch_loss = {}
                    # tmp_time = time.time()
                    for j in range(self.num_ensemble):
                        data_this_ens = to_torch(data_ensemble[j], dtype='float32', device=self.device, non_blocking=True)
                        # Extract ids of the environment
                        if self.add_ori:
                            if "id" in data_this_ens["obs"]:
                                ids = data_this_ens["obs"].pop("id")
                            else: raise "Not found the ids of the cabinets or the doors!" 
                        
                        for key in data_this_ens:
                            if not isinstance(data_this_ens[key], dict) and data_this_ens[key].ndim == 1:
                                data_this_ens[key] = data_this_ens[key][..., None]

                        outputs = self.model(data_this_ens["obs"], data_this_ens["actions"], ids, j) 
                    
                        # Calculate loss and back propagate
                        ground_truth = self.get_ground_truth(data_this_ens)
                        ensemble_loss = self.calculate_loss(outputs, ground_truth)
                        self.model_optim.zero_grad()
                        ensemble_loss['total'].backward()
                        self.model_optim.step()
                        batch_loss = add_dict_array(batch_loss, ensemble_loss)
                    # batch_time = time.time() - tmp_time
                    # print("batch_time: ", batch_time)
                    for each_loss in batch_loss.values(): each_loss /= self.num_ensemble * num_batch
                    epoch_train_loss = add_dict_array(epoch_train_loss, batch_loss)

                for each_loss in epoch_train_loss.values(): each_loss /= num_batch
                train_loss.append((self.epoch_counter, epoch_train_loss))

                # assert len(trainOnPol_batches_copy) == len(trainOnPol_batches)
                # for i in range(len(trainOnPol_batches)):
                #     assert compare_dict_array(trainOnPol_batches[i], trainOnPol_batches_copy[i])
                # assert len(trainRand_batches_copy) == len(trainRand_batches)
                # for i in range(len(trainRand_batches)):
                #     assert compare_dict_array(trainRand_batches[i], trainRand_batches_copy[i])

                # Validate the model
                epoch_valRand_loss, epoch_valOnPol_loss = dict(), dict()
                # Validation loss on random set
                valRand_batches = valRand.get_all()
                valRand_data = to_torch(valRand_batches, dtype='float32', device=self.device, non_blocking=True)
                if self.add_ori: ids = valRand_data["obs"].pop("id")
                ground_truth = self.get_ground_truth(valRand_data)
                with torch.no_grad():
                    valRand_outputs = concat(self.model(valRand_data["obs"], valRand_data["actions"], ids), axis=0)
                    epoch_valRand_loss = add_dict_array(epoch_valRand_loss, 
                        self.calculate_loss(valRand_outputs, ground_truth))
                    for each_loss in epoch_valRand_loss.values(): each_loss /= self.num_ensemble
                valRand_loss.append((self.epoch_counter, epoch_valRand_loss))

                # Validation loss on on-policy set
                valOnPol_batches = valOnPol.get_all()
                valOnPol_data = to_torch(valOnPol_batches, dtype='float32', device=self.device, non_blocking=True)
                if self.add_ori: ids = valOnPol_data["obs"].pop("id")
                ground_truth = self.get_ground_truth(valOnPol_data)
                with torch.no_grad():
                    valOnPol_outputs = concat(self.model(valOnPol_data["obs"], valOnPol_data["actions"], ids), axis=0)
                    epoch_valOnPol_loss = add_dict_array(epoch_valOnPol_loss, 
                        self.calculate_loss(valOnPol_outputs, ground_truth))
                    for each_loss in epoch_valOnPol_loss.values(): each_loss /= self.num_ensemble
                valOnPol_loss.append((self.epoch_counter, epoch_valOnPol_loss))

        elif isinstance(self.nstep, int) and self.nEpoch is None:
            raise NotImplementedError
        else: 
            raise TypeError("Cannot determine how many times the model is updated when training.") 

        return train_loss, valOnPol_loss, valRand_loss

    def get_ground_truth(self, data, start=0, end=None):
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

    def calculate_loss(self, pred, ground_truth):
        ChamLoss = chamfer_3DDist()
        loss_dr = F.mse_loss(ground_truth['rewards'].squeeze(), pred['rewards'])
        loss_ds = F.mse_loss(ground_truth['ds'], pred['ds'])
        dist1, dist2, _, _ = ChamLoss(ground_truth['next_pcd'], pred['dp'] + ground_truth['pointcloud'])
        chamfer = torch.mean(dist1) + torch.mean(dist2)
        total_loss = loss_dr * self.loss_coeff['rewards'] + loss_ds * self.loss_coeff['ds'] 
        total_loss += chamfer * self.loss_coeff['chamfer']
        loss_dict = {"total": total_loss, "rewards": loss_dr, "ds": loss_ds, "next_pcd": chamfer}

        return loss_dict

    def update_norm(self, reward_mean, reward_std, ds_mean, ds_std):
        self.policy.update_norm(reward_mean, reward_std, ds_mean, ds_std)

    def reset(self, *args):
        self.policy.reset(*args)