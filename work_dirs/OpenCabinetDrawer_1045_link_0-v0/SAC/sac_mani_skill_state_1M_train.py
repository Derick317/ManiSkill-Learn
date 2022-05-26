agent = dict(
    type='SAC',
    batch_size=1024,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    alpha_optim_cfg=dict(type='Adam', lr=0.0003),
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead', log_sig_min=-20, log_sig_max=2,
            epsilon=1e-06),
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=['obs_shape', 256, 256, 256, 'action_shape * 2'],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        optim_cfg=dict(type='Adam', lr=0.0003)),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            bias='auto',
            mlp_spec=['obs_shape + action_shape', 256, 256, 256, 1],
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        optim_cfg=dict(type='Adam', lr=0.0003)))
log_level = 'INFO'
train_mfrl_cfg = dict(
    on_policy=False,
    total_steps=1000000,
    warm_steps=4000,
    n_eval=200000,
    n_checkpoint=200000,
    n_steps=8,
    n_updates=4)
rollout_cfg = dict(
    type='BatchRollout',
    use_cost=False,
    reward_only=False,
    num_procs=8,
    with_info=False,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        reward_type='dense',
        env_name='OpenCabinetDrawer_1045_link_0-v0'))
eval_cfg = dict(
    type='BatchEvaluation',
    num=10,
    num_procs=2,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
    env_cfg=dict(
        type='gym',
        unwrapped=False,
        reward_type='dense',
        env_name='OpenCabinetDrawer_1045_link_0-v0'))
env_cfg = dict(
    type='gym',
    unwrapped=False,
    reward_type='dense',
    env_name='OpenCabinetDrawer_1045_link_0-v0')
replay_cfg = dict(type='ReplayMemory', capacity=1000000)
work_dir = './work_dirs/OpenCabinetDrawer_1045_link_0-v0/SAC'
