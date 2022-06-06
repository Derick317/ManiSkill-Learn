log_level = 'INFO'


rollout_cfg = dict(
    type='Rollout',
    use_cost=False,
    reward_only=False,
    num_procs=1,
)


eval_cfg = dict(
    type='Evaluation',
    num=10,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
)