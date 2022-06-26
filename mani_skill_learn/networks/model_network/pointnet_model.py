import torch
import torch.nn as nn
import numpy as np

from mani_skill_learn.utils.torch import ExtendedModule
from ..builder import MODELNETWORKS, build_backbone
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape
from ...utils.oracle import door_orientation as door_ori
from ...utils.data import to_np, to_torch
from ...utils.math import split_num

class PointNetModelBackbone(ExtendedModule):
    def __init__(self, nn_cfg):
        super(PointNetModelBackbone, self).__init__()
        pointnet_cfg = nn_cfg['pointnet_cfg']
        feat_flow_cfg = nn_cfg['feat_flow_cfg']
        reward_state_cfg = nn_cfg['reward_state_cfg']

        assert len(pointnet_cfg.mlp_cfg.mlp_spec) == 0, "Unexpected neural network after pooling!"

        self.pointnet = build_backbone(pointnet_cfg)
        self.feat_to_flow = build_backbone(feat_flow_cfg)
        self.reward_state = build_backbone(reward_state_cfg)

    def forward(self, pcd, state, action, mask=None):
        state = torch.cat([state, action], dim=-1)
        global_feat, point_feat = self.pointnet(pcd, state, mask)
        N = point_feat.shape[1] # the number of points
        point_global = torch.cat([point_feat, global_feat[:, None, :].repeat(1, N, 1)], dim=-1)
        dp = self.feat_to_flow(point_global.transpose(2, 1)).transpose(2, 1)
        reward_and_state = self.reward_state(global_feat)
        re = reward_and_state[..., 0]
        ds = reward_and_state[..., 1:]

        return dp, re, ds

@MODELNETWORKS.register_module()
class PointNetModel(ExtendedModule):
    def __init__(self, 
                 nn_cfg,
                 del_rgb=False,
                 add_ori=False, 
                 obs_shape=None, 
                 action_shape=None, 
                 num_ensemble=1):
        super(PointNetModel, self).__init__()

        self.reward_mean = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.reward_std = nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.ds_mean = nn.Parameter(torch.zeros(obs_shape['state']), requires_grad=False)
        self.ds_std = nn.Parameter(torch.ones(obs_shape['state']), requires_grad=False)

        self.del_rgb = del_rgb
        self.add_ori = add_ori # Whether to add the orientation of the door to the input
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        if add_ori:
            nn_cfg.pointnet_cfg.conv_cfg.mlp_spec[0] += 1
        self.ensembles = nn.ModuleList()
        self.num_ensemble = num_ensemble
        for _ in range(num_ensemble):
            self.ensembles.append(PointNetModelBackbone(nn_cfg))

    def init_weights(self, pretrained=None, init_cfg=None):
        if not isinstance(pretrained, (tuple, list)):
            pretrained = [pretrained for i in range(len(self.ensembles))]
        for i in range(len(self.ensembles)):
            self.ensembles[i].init_weights(pretrained[i], **init_cfg)

    def forward(self, obs, action, ids=None, ensemble_id=None):
        state = obs["state"]
        pcd = obs["pointcloud"]
        
        # Process the dimension
        if len(state.shape) == len(ids.shape) == len(action.shape) == 2 and len(pcd['xyz'].shape) == 3:
            M = 0
        elif len(state.shape) == len(ids.shape) == len(action.shape) == 3 and len(pcd['xyz'].shape) == 4:
            assert state.shape[0] == ids.shape[0] == pcd['xyz'].shape[0] == action.shape[0]
            M = state.shape[0]
            state = state.view(-1, state.shape[2])
            ids = ids.view(-1, ids.shape[2])
            action = action.view(-1, action.shape[2])
            for key in pcd.keys():
                pcd[key] = pcd[key].view(-1, pcd[key].shape[2], pcd[key].shape[3])
        else: raise NotImplementedError

        if self.del_rgb and 'rgb' in pcd: del pcd['rgb']
        if self.add_ori:
            if ids is None:
                raise "Not found the id of the cabinet or the door!"
            B = state.shape[0]

            assert len(state.shape) == 2 and B == ids.shape[0]
            ori = np.ones((B, 1))
            for j in range(B):
                if door_ori[str(int(ids[j][0]))][int(ids[j][1])] == 'l': ori[j][0] = -1
            ori = to_torch(ori, dtype='float32', device=state.device, non_blocking=True)
            state = torch.cat([state, ori], dim=-1)
        if ensemble_id == None:
            _, len_ensemble = split_num(B, self.num_ensemble)
            output = []
            start = 0
            for i in range(self.num_ensemble):
                ens_pcd = dict()
                for key in pcd:
                    ens_pcd[key] = pcd[key][start: start + len_ensemble[i]]
                ens_state = state[start: start + len_ensemble[i]]
                ens_act = action[start: start + len_ensemble[i]]
                start += len_ensemble[i]
                dp, re, ds = self.ensembles[i](ens_pcd, ens_state, ens_act)
                output.append({"dp": dp, "rewards": re, "ds": ds})
        else:
            dp, re, ds = self.ensembles[ensemble_id](pcd, state, action)
            output = {"dp": dp, "rewards": re, "ds": ds}
        
        if M > 0:
            output['dp'] = output['dp'].view(M, -1, output['dp'].shape[1], output['dp'].shape[2])
            output['rewards'] = output['rewards'].view(M, -1)
            output['ds'] = output['ds'].view(M, -1, output['ds'].shape[1])
            state = state.view(M, -1, state.shape[1])
            ids = ids.view(M, -1, ids.shape[1])
            action = action.view(M, -1, action.shape[1])
            for key in pcd.keys():
                pcd[key] = pcd[key].view(M, -1, pcd[key].shape[1], pcd[key].shape[2])
        return output

    def update_norm(self, reward_mean, reward_std, ds_mean, ds_std):
        reward_mean = to_torch(reward_mean, dtype='float32', device=self.device, non_blocking=True)
        reward_std = to_torch(reward_std, dtype='float32', device=self.device, non_blocking=True)
        ds_mean = to_torch(ds_mean, dtype='float32', device=self.device, non_blocking=True)
        ds_std = to_torch(ds_std, dtype='float32', device=self.device, non_blocking=True)

        self.reward_mean = nn.Parameter(reward_mean, requires_grad=False)
        self.reward_std = nn.Parameter(reward_std, requires_grad=False)
        self.ds_mean = nn.Parameter(ds_mean, requires_grad=False)
        self.ds_std = nn.Parameter(ds_std, requires_grad=False)