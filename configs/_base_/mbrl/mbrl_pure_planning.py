_base_ = ['./mbrl.py']

env_cfg = dict(
    type='gym',
    unwrapped=False,
    max_episode_steps=200,
    obs_mode='state',
    reward_type='dense',
)

agent = dict(
    type='PurePlanning',
    num_procs=10,
    policy_cfg=dict(
        horizon=5,
        num_action_sequences=25,
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
    )
)

replay_cfg = dict(
    type='ReplayMemory',
    capacity=1000000,
)

rollout_cfg = dict(
    type='Rollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=1,
)

eval_cfg = dict(
    type='Evaluation',
    num=1,
    num_procs=1,
    use_hidden_state=True,
    start_state=None,
    save_traj=True,
    save_log=True,
    use_tf_log=True,
    save_video=True,
    use_log=True,
    log_every_step=True
)
