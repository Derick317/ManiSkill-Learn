python -m tools.run_rl configs/mbrl/mani_skill_pure_planning.py \
--seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1033_link_1-v0" \
--work-dir=./work_dirs/test_planning/mppi/1033 \
--gpu-ids=0 \
--evaluation