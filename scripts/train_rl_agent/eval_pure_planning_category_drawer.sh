#!/bin/bash  

cab_id=(00 04 05 13 16 21 24 27 32 33 35 38 40 44 45 52 54 56 61 63 66 67 76 79 82)
drawer_num=(2 2 4 1 4 1 3 2 1 3 1 1 1 2 1 1 1 3 1 1 1 1 4 4 3)

for((i=0; i<25; i=i+1))
do
    cabinet_name="10${cab_id[i]}"
    iter_n=`expr 12 / ${drawer_num[i]}`
    for((j=0; j<${drawer_num[i]}; j=j+1))
    do
        for((k=0; k<${iter_n}; k=k+1))
        do
            python -m tools.run_rl configs/mbrl/mani_skill_pure_planning.py \
            --seed=0 \
            --cfg-options "env_cfg.env_name=OpenCabinetDrawer_${cabinet_name}_link_${j}-v0" \
            --work-dir=work_dirs/test_planning/mppi/category \
            --gpu-ids=0 \
            --evaluation

            mv "work_dirs/test_planning/mppi/category/PurePlanning" "work_dirs/test_planning/mppi/category/${cabinet_name}_${j}(${k})"
        done
    done
done