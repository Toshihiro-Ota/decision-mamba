
# Atari

We build our Atari implementation on top of [minGPT](https://github.com/karpathy/minGPT) and benchmark our results on the [DQN-replay](https://github.com/google-research/batch_rl) dataset.

## Installation

Dependencies can be installed with the following:

```bash
pip install -r requirements.txt
```

## Downloading datasets

Create a directory for dataset and load the datasets using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)

```bash
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

```bash
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar

python -m atari_py.import_roms ROMS
```

## Example usage

To train a model, e.g. DMamba, run the following:

```bash
python train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix /path/to/[DIRECTORY_NAME]/ \
        --context_length 30 \
        --n_layer 6 \
        --n_embd 128 \
        --token_mixer 'mamba' \
        --epochs 10 \
        --batch_size 256 \
        --num_steps 500000 \
        --num_buffers 50 \
        --trajectories_per_buffer 10 \
        --output /path/to/output \
        --experiment dmamba_breakout \
        --seed 123
```

Adding `-w True` or `--log_to_mlflow True` will log results.
Script to train for other datasets can be found in `run_atari.sh`.
