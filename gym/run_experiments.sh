#!/bin/bash

DATA_DIR=/data/data_gym/
OUT_DIR=/output/gym


EXP_H=dmamba_hopper
for seed in 12 23 34 45 51; do python experiment.py --env 'hopper' --dataset 'medium' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 256 --learning_rate 1e-4 --output $OUT_DIR --experiment "$EXP_H"_medium --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'hopper' --dataset 'medium-replay' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 256 --learning_rate 1e-4 --output $OUT_DIR --experiment "$EXP_H"_medium-replay --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'hopper' --dataset 'medium-expert' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_H"_medium-expert --seed $seed; done

EXP_C=dmamba_halfcheetah
for seed in 12 23 34 45 51; do python experiment.py --env 'halfcheetah' --dataset 'medium' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_C"_medium --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'halfcheetah' --dataset 'medium-replay' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_C"_medium-replay --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'halfcheetah' --dataset 'medium-expert' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_C"_medium-expert --seed $seed; done

EXP_W=dmamba_walker2d
for seed in 12 23 34 45 51; do python experiment.py --env 'walker2d' --dataset 'medium' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-4 --output $OUT_DIR --experiment "$EXP_W"_medium --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'walker2d' --dataset 'medium-replay' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_W"_medium-replay --seed $seed; done
for seed in 12 23 34 45 51; do python experiment.py --env 'walker2d' --dataset 'medium-expert' --data_dir_prefix $DATA_DIR --model_type 'dmamba' --embed_dim 128 --learning_rate 1e-3 --output $OUT_DIR --experiment "$EXP_W"_medium-expert --seed $seed; done


#EXP_T=dt_hopper_medium
#python experiment.py --env 'hopper' --dataset 'medium' --data_dir_prefix $DATA_DIR --K 20 --model_type 'dt' --embed_dim 256 --output $OUT_DIR --experiment $EXP_T --seed 1
