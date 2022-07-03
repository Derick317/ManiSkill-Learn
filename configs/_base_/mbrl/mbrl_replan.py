_base_ = ['./mbrl.py']

stack_frame = 1

env_cfg = dict(
    type='gym',
    unwrapped=False,
    env_name="OpenCabinetDoor-v0",
    max_episode_steps=200,
    obs_mode='pointcloud',
    reward_type='dense',
    id_in_obs=True
)

agent = dict(
    type='REPLANV2',
    del_rgb=True,
    batchsize=128,
    add_ori=True,
    num_ensemble=3,
    nEpoch=24, # 40
    loss_coeff = {'rewards': 1, 'ds': 1, 'chamfer': 100},
    policy_cfg=dict(
        horizon=7,
        num_action_sequences=125,
        noise_std=0.04,
        sample_strategy='mppi', # only 'mppi', 'cem' and 'random' for now
        cem_cfg=dict(
            cem_iterations=4,
            cem_num_elites=10,
            cem_alpha=1.0
        ),
        mppi_cfg=dict(
            mppi_gamma=4,
            mppi_beta=0.7,
            sample_velocity=True,
            mag_noise=0.9,
        ),
    ),
    rand_policy_cfg=dict(
        type="RandPolicy",
        sample_velocities=False,
        vel_min = 0.05,
        vel_max = 0.15,
        hold_action = 2
    ),
    model_cfg=dict(
        type="PointNetModel",
        optim_cfg=dict(
            type='Adam', 
            lr=5e-4, 
            weight_decay=5e-6
        ),
        nn_cfg=dict(
            pointnet_cfg=dict(
                type="PointNetV1",
                stack_frame = stack_frame,
                conv_cfg=dict(
                    type="ConvMLP",
                    mlp_spec=['agent_shape + action_shape + pcd_xyz_seg_channel', 128, 256, 1024],
                    inactivated_output=False,
                    conv_init_cfg=dict(
                        type='xavier_init',
                        gain=1,
                        bias=0,
                    ),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    mlp_spec=[],
                ),
            ),
            feat_flow_cfg=dict(
                type="ConvMLP",
                mlp_spec=['128 + 1024', 256, 3],
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            reward_state_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['1024', 256, 'agent_shape + 1'],
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            )
        )
    )
)

train_mbrl_cfg=dict(
    n_traj_onPol=24,
    n_traj_rand=3,
    total_steps=2500000,
    warm_steps=4000,
    n_eval=50000,
    n_checkpoint=25000,
    reset_hp=True,
    on_policy=False
)

rollout_cfg = dict(
    type='Rollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=3,
)

eval_cfg = dict(
    type='Evaluation',
    num=5,
    num_procs=1,
    use_hidden_state=False, # Use hidden state is only for CEM ground-truth model evaluation
    start_state=None,
    enable_merge=False,
    save_traj=True,
    save_log=False,
    use_tf_log=False,
    save_video=True,
    use_log=False,
    log_every_step=False,
    env_cfg = dict(
        type='gym',
        unwrapped=False,
        env_name="OpenCabinetDoor-v0",
        max_episode_steps=200,
        obs_mode='pointcloud',
        reward_type='dense',
        id_in_obs=True
    )
)

replay_train_rand_cfg=dict(
    type='ReplayMemory',
    capacity=160000,
)

replay_train_onPol_cfg=dict(
    type='ReplayMemory',
    capacity=600000,
)

replay_val_rand_cfg=dict(
    type='ReplayMemory',
    capacity=100000,
)

replay_val_onPol_cfg=dict(
    type='ReplayMemory',
    capacity=100000,
)
