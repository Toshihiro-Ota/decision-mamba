import os
import argparse
from datetime import datetime
import logging
import yaml
import torch

from mingpt.utils import set_seed, get_outdir
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from data_atari.create_dataset import create_dataset, StateActionReturnDataset


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    import mlflow
    has_mlflow = True
except ImportError:
    has_mlflow = False


parser = argparse.ArgumentParser()
# Dataset parameters
parser.add_argument('--game', type=str, default='Breakout',
                    help='Name of game (default: "Breakout")')
parser.add_argument('--data_dir_prefix', type=str, default='data-atari/',
                    help='Path to dataset (default: "data-atari/", must include the last slash "/")')
parser.add_argument('--context_length', type=int, default=30,
                    help='Context length K (default: 30)')

# Model parameters
parser.add_argument('--model_type', type=str, default='reward_conditioned',
                    help='Set "reward_conditioned" or "naive" (default: "reward_conditioned")')
parser.add_argument('--n_layer', type=int, default=6,
                    help='Number of layers of the model (default: 6)')
parser.add_argument('--n_embd', type=int, default=128,
                    help='Embedding dim of tokens (default: 128)')
parser.add_argument('--token_mixer', type=str, default='mamba',
                    help='Choose one of "attn", "conv", "conv-attn", "mamba-min" or "mamba" (default: "mamba")')
# Model-specific parameters
parser.add_argument('--n_head', type=int, default=8,
                    help='Number of heads for "attn" (default: 8)')
parser.add_argument('--conv_window_size', type=int, default=6,
                    help='Conv window size for "conv" (default: 6)')
parser.add_argument('--conv_proj', action='store_true',
                    help='An FC layer follows conv')

# Training parameters
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Input batch size for training (default: 256)')
parser.add_argument('--learning_rate', '-lr', type=float, default=6e-4,
                    help='Base learning rate (default: 6e-4)')
parser.add_argument('--num_steps', type=int, default=500000,
                    help='Number of steps for training, roughly the dataset size (default: 500000)')
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                    help='Number of trajectories to sample from each of the buffers (default: 10)')

# Misc
parser.add_argument('--seed', type=int, default=123,
                    help='Random seed (default: 123)')
parser.add_argument('--output', type=str, default='',
                    help='Path to output folder (default: none, current dir)')
parser.add_argument('--experiment', type=str, default='',
                    help='Name of training experiment, name of sub-folder for output (default: none)')
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,
                    help='Log training and validation metrics to wandb')
parser.add_argument('--log_to_mlflow', type=bool, default=False,
                    help='Log training and validation metrics to mlflow')


def main():
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    args = parser.parse_args()

    if args.log_to_wandb:
        if has_wandb:
            wandb.init(
                name=f'{args.game}-{args.seed}-{args.token_mixer}',
                group=f'{args.token_mixer}',
                project='[Decision Mamba] Atari',
                config=args,
            )
        else:
            print("You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    set_seed(args.seed)

    # setup output dir and yaml file
    output_dir = None
    save_model_name = args.token_mixer + "_" + args.game.lower() + ".cpt"
    if args.experiment:
        exp_name = args.experiment + str(args.seed)
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.token_mixer
            ])
    output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    args_dir = os.path.join(output_dir, 'args.yaml')
    with open(args_dir, 'w') as f:
        f.write(yaml.safe_dump(args.__dict__, default_flow_style=False))

    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers,
                                                                        args.num_steps,
                                                                        args.game,
                                                                        args.data_dir_prefix,
                                                                        args.trajectories_per_buffer,
                                                                        )
    train_dataset = StateActionReturnDataset(obss,
                                            args.context_length*3,
                                            actions,
                                            done_idxs,
                                            rtgs,
                                            timesteps,
                                            )

    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        model_type=args.model_type,
        n_embd=args.n_embd,
        token_mixer=args.token_mixer,
        window_size=args.conv_window_size,
        conv_proj=args.conv_proj,
        max_timestep=max(timesteps),
        )
    model = GPT(mconf)

    tconf = TrainerConfig(
        game=args.game,
        model_type=args.model_type,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=2*len(train_dataset)*args.context_length*3,
        num_workers=2,
        seed=args.seed,
        max_timestep=max(timesteps),
        log_to_wandb=args.log_to_wandb,
        log_to_mlflow=args.log_to_mlflow,
        output_dir=output_dir,
        args_dir=args_dir,
        )
    trainer = Trainer(model, train_dataset, None, tconf)

    try:
        trainer.train()
    except KeyboardInterrupt:
        pass

    save_state = {
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        #'optimizer': optimizer.state_dict(),
    }
    torch.save(save_state, os.path.join(output_dir, save_model_name))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.log_to_mlflow and has_mlflow:
        mlflow.set_experiment(os.path.basename(args.output))
        with mlflow.start_run(run_name=args.experiment + "_" + str(args.seed)):
            mlflow.log_artifact('./train_atari.py')
            mlflow.log_params({k: v for k, v in vars(args).items()})
            main()
    else:
        main()
