python -m tools.run_rl configs/mbrl/mani_skill_pure_planning.py \
--seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" \
--work-dir=./work_dirs/test_planning/cem \
--gpu-ids=0 \
--evaluation