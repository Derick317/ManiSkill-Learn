from ..builder import MBRL
from ...env.env_utils import build_env, true_done
import numpy as np
from mani_skill_learn.utils.torch import BaseAgent
import time, math, copy

@MBRL.register_module()
class REPLAN(BaseAgent):
    def __init__(self, policy_cfg, model_cfg, obs_shape, action_shape, action_space):
        super(REPLAN, self).__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_space = action_space
        self.policy = MPCPolicy(ac_dim=action_shape, ac_space=action_space, **policy_cfg)

class MPCPolicy():
    pass