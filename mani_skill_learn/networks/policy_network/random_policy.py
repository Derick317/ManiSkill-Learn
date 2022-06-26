from ..builder import POLICYNETWORKS
import numpy as np

@POLICYNETWORKS.register_module()
class RandPolicy():
    def __init__(self, ac_space, policy_cfg) -> None:
        self.ac_shape = ac_space
        self.low = ac_space.low
        self.high = ac_space.high
        self.shape = ac_space.low.shape
        self.sample_velocities = policy_cfg['sample_velocities']
        self.vel_min = policy_cfg['vel_min']
        self.vel_max = policy_cfg['vel_max']
        self.hold_action = policy_cfg['hold_action']
        self.device = 'cpu'

        self.counter = 0

    # get_action
    def __call__(self, obs, prev_action):        
        # for a position-controlled robot, sample random velocities instead of random actions for smoother exploration
        if self.sample_velocities:
            if prev_action is None:
                # generate random action for right now
                self.rand_ac = np.random.uniform(self.low, self.high, self.shape)
                action = self.rand_ac

                # generate velocity, to be used if next steps might hold_action
                self.vel_sample = np.random.uniform(self.vel_min, self.vel_max, self.shape)
                self.direction_num = np.random.randint(0, 2, self.shape)
                self.vel_sample[self.direction_num == 0] = -self.vel_sample[self.direction_num == 0]
            else:
                if (self.counter % self.hold_action) == 0:
                    self.vel_sample = np.random.uniform(self.vel_min, self.vel_max, self.shape)
                    self.direction_num = np.random.randint(0, 2, self.shape)
                    self.vel_sample[self.direction_num == 0] = -self.vel_sample[self.direction_num == 0]

                    # go opposite direction if you hit limit
                    self.vel_sample[prev_action <= self.low] = np.abs(self.vel_sample)[prev_action <= self.low] # need to do larger action
                    self.vel_sample[prev_action >= self.high] = -np.abs(self.vel_sample)[prev_action >= self.high]
                # new action
                action = prev_action + self.vel_sample

        # else, for a torque-controlled robot, just uniformly sample random actions
        else:
            if (self.counter % self.hold_action) == 0:
                self.rand_ac = np.random.uniform(self.low, self.high, self.shape)
            action = self.rand_ac

        self.counter += 1

        return action