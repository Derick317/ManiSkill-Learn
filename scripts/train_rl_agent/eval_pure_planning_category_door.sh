#!/bin/bash  

cab_id=(00 01 02 06 07 \
        14 17 18 25 26 \
        27 28 30 31 34 \
        36 38 39 41 42 \
        44 45 46 47 49 \
        51 52 54 57 60 \
        61 62 63 64 65 \
        67 68 73 75 77 \
        78 81)
door_num=(2 1 1 1 2 \
          1 1 2 2 1 \
          1 1 1 2 2 \
          2 1 2 1 4 \
          4 1 1 1 1 \
          1 2 1 1 4 \
          1 1 1 4 2 \
          1 1 1 2 1 \
          2 1)

# sum=0
# for((i=0; i<42; i=i+1))
# do
#     sum=`expr ${sum} + ${door_num[i]}`
# done
# echo ${sum}

for((i=25; i<42; i=i+1))
do
    cabinet_name="10${cab_id[i]}"
    iter_n=`expr 8 / ${door_num[i]}`
    mkdir work_dirs/test_planning/mppi_door/category/${cabinet_name}
    for((j=0; j<${door_num[i]}; j=j+1))
    do
        for((k=0; k<${iter_n}; k=k+1))
        do
            python -m tools.run_rl configs/mbrl/mani_skill_pure_planning.py \
            --seed=0 \
            --cfg-options "env_cfg.env_name=OpenCabinetDoor_${cabinet_name}_link_${j}-v0" \
            --work-dir=work_dirs/test_planning/mppi_door/category \
            --gpu-ids=0 \
            --evaluation

            mv "work_dirs/test_planning/mppi_door/category/PurePlanning" "work_dirs/test_planning/mppi_door/category/${cabinet_name}/link_${j}(${k})"
        done
    done
done