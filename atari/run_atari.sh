#!/bin/bash

DATA_DIR=/data/data_atari/
OUT_DIR=/output/atari


EXP_B=dmamba_breakout
for seed in 123 231 312; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_B --seed $seed; done

EXP_Q=dmamba_qbert
for seed in 123 231 312; do python train_atari.py --game 'Qbert' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

EXP_P=dmamba_pong
for seed in 123 231 312; do python train_atari.py --game 'Pong' --data_dir_prefix $DATA_DIR --context_length 50 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_P --seed $seed; done

EXP_S=dmamba_seaquest
for seed in 123 231 312; do python train_atari.py --game 'Seaquest' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_S --seed $seed; done


#EXP_T=dt_breakout
#python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_T --seed 1
#EXP_C=dc_breakout
#python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 8 --token_mixer 'conv' --output $OUT_DIR --experiment $EXP_C --seed 1
