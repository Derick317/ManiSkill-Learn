_base_ = ['./mbrl.py']

stack_frame = 1

env_cfg = dict(
    type='gym',
    unwrapped=False,
    env_name="OpenCabinetDoor_Split_Train-v0",
    max_episode_steps=200,
    obs_mode='pointcloud',
    reward_type='dense',
    id_in_obs=True
)

agent = dict(
    type='REPLAN',
    del_rgb=True,
    batchsize=256,
    add_ori=True,
    num_ensemble=3,
    nEpoch=25, # 40
    loss_coeff = {'rewards': 1, 'ds': 1, 'chamfer': 100},
    policy_cfg=dict(
        horizon=5,
        num_action_sequences=25,
        noise_std=0.1,
        sample_strategy='mppi', # only 'mppi', 'cem' and 'random' for now
        cem_cfg=dict(
            cem_iterations=4,
            cem_num_elites=10,
            cem_alpha=1.0
        ),
        mppi_cfg=dict(
            mppi_gamma=4,
            mppi_beta=0.3,
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
            weight_decay=1e-5
        ),
        nn_cfg=dict(
            pointnet_cfg=dict(
                type="PointNetV1",
                stack_frame = stack_frame,
                conv_cfg=dict(
                    type="ConvMLP",
                    mlp_spec=['agent_shape + action_shape + pcd_xyz_seg_channel', 128, 512],
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
                mlp_spec=['128 + 512', 128, 3],
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            reward_state_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['512', 128, 'agent_shape + 1'],
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
    n_traj_onPol=25,
    n_traj_rand=5,
    total_steps=2500000,
    warm_steps=4000,
    n_eval=100000,
    n_checkpoint=100000,
    on_policy=False
)

rollout_cfg = dict(
    type='Rollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=5,
)

eval_cfg = dict(
    type='Evaluation',
    num=72,
    num_procs=4,
    use_hidden_state=False, # Use hidden state is only for CEM ground-truth model evaluation
    start_state=None,
    enable_merge=False,
    save_traj=True,
    save_log=False,
    use_tf_log=True,
    save_video=True,
    use_log=True,
    log_every_step=False,
    env_cfg = dict(
        type='gym',
        unwrapped=False,
        env_name="OpenCabinetDoor_Split_Val-v0",
        max_episode_steps=200,
        obs_mode='pointcloud',
        reward_type='dense',
        iter_all_cabinet=True,
        id_in_obs=True
    )
)

replay_train_rand_cfg=dict(
    type='ReplayMemory',
    capacity=160000,
)

replay_train_onPol_cfg=dict(
    type='ReplayMemory',
    capacity=800000,
)

replay_val_rand_cfg=dict(
    type='ReplayMemory',
    capacity=100000,
)

replay_val_onPol_cfg=dict(
    type='ReplayMemory',
    capacity=100000,
)
