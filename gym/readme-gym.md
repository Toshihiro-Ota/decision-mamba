
# OpenAI Gym

## Installation

Experiments require MuJoCo. Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
For example, you can do that by running the following commands:

```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz

mkdir ~/.mujoco
cp -r mujoco210 ~/.mujoco/mujoco210
```

Then, dependencies can be installed with the following:

```bash
pip install -r requirements.txt
```

If you get dependency conflicts among `transformers`, `tokenizers` and `mamba_ssm`, you can do:

- Manually install `transformers` by running `git clone https://github.com/huggingface/transformers.git -b v4.11.0 transformers-4.11.0`, replace `"tokenizers>=0.10.1,<0.11",` of `transformers-4.11.0/setup.py#L151` to `"tokenizers>=0.10.1",` and run `make deps_table_update` before `pip install transformers-4.11.0`
- Comment out or remove `from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel` of `mamba_ssm/__init__.py#L5`

## Downloading datasets

Create a directory for dataset, e.g. `data`. Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
You need set up your machine first, like so:

```bash
sudo apt update
sudo apt install gcc

sudo apt build-dep mesa
sudo apt install llvm-dev
sudo apt install freeglut3 freeglut3-dev

sudo apt install build-essential python3-dev

sudo apt install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libglfw3-dev libosmesa6-dev patchelf
```

```bash
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```

Then after rebooting the machine, add an environment variable as follows. The appropriate path is originally stored in `LD_LIBRARY_PATH`, which might be different from like `/usr/lib/nvidia` below.

```bash
echo $LD_LIBRARY_PATH
env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

Finally, run the following script in order to download the datasets and save them in our format:

```bash
python ./util/download_d4rl_datasets.py --suite "locomotion" --data-dir /path/to/data/
```

## Example usage

Experiments for our Decision Mamba can be reproduced with the following:

```bash
python experiment.py \
        --env 'hopper' \
        --dataset 'medium' \
        --data_dir_prefix /path/to/data/ \
        --K 20 \
        --model_type 'dmamba' \
        --n_layer 3 \
        --embed_dim 256 \
        --activation_function 'gelu' \
        --max_iters 10 \
        --batch_size 64 \
        --num_steps_per_iter 10000 \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --num_eval_episodes 100 \
        --output /path/to/output \
        --experiment dmamba_hopper_medium \
        --seed 123
```

Adding `-w True` or `--log_to_mlflow True` will log results.
