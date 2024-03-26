import os
import argparse
from datetime import datetime
import random
import pickle
import yaml

import numpy as np
import torch
from d4rl import infos

from util.trainer import Trainer, get_env_info, evaluate_episode_rtg, get_model_optimizer
from util.utils import set_seed, discount_cumsum, get_outdir, update_summary


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
parser.add_argument('--env', type=str, default='hopper',
                    help='Name of Env (default: "hopper")')
parser.add_argument('--dataset', type=str, default='medium',
                    help='Choose one of "medium", "medium-replay", "medium-expert", or "expert" (default: "medium")')
parser.add_argument('--data_dir_prefix', type=str, default='data-gym/',
                    help='Path to dataset (default: "data-gym/", must include the last slash "/")')
parser.add_argument('--mode', type=str, default='normal',
                    help='"normal" for standard setting, "delayed" for sparse, no-reward-decay (default: "normal")')
parser.add_argument('--K', type=int, default=20,
                    help='Context length K (default: 20)')
parser.add_argument('--pct_traj', type=float, default=1.,
                    help='Only train on top pct_traj trajectories for %%BC experiment (default: 1.)')

# Model parameters
parser.add_argument('--model_type', type=str, default='dmamba',
                    help='Choose one of "dt", "dmamba-min" or "dmamba" (default: "dmamba")')
parser.add_argument('--n_layer', type=int, default=3,
                    help='Number of layers of the model (default: 3)')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='Embedding dim of tokens (default: 128)')
parser.add_argument('--activation_function', type=str, default='gelu',
                    help='Activation function for the MLP layer (default: "gelu")')
parser.add_argument('--dropout', type=float, default=0.1)
# Model-specific parameters
parser.add_argument('--n_head', type=int, default=1,
                    help='Number of heads for "dt" (default: 1)')
parser.add_argument('--conv_window_size', type=int, default=6,
                    help='Conv window size for "dc" (default: 6)')

# Training parameters
parser.add_argument('--max_iters', type=int, default=10,
                    help='Max number of iterations for training (default: 10)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Input batch size for training (default: 64)')
parser.add_argument('--num_steps_per_iter', type=int, default=10000,
                    help='Number of training steps for one iteration (default: 10000)')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
                    help='Base learning rate (default: 1e-3)')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4,
                    help='Weight decay (default: 1e-4)')
parser.add_argument('--warmup_steps', type=int, default=10000,
                    help='Steps to warmup LR, if scheduler supports (default: 10000)')
parser.add_argument('--num_eval_episodes', type=int, default=100,
                    help='Number of episodes for evaluation (default: 100)')
parser.add_argument('--remove_act_embs', action='store_true',
                    help='Remove action embeddings')

# Misc
parser.add_argument('--seed', type=int, default=123,
                    help='Random seed (default: 123)')
parser.add_argument('--output', type=str, default='',
                    help='Path to output folder (default: none, current dir)')
parser.add_argument('--experiment', type=str, default='',
                    help='Name of train experiment, name of sub-folder for output (default: none)')
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,
                    help='Log training and validation metrics to wandb')
parser.add_argument('--log_to_mlflow', type=bool, default=False,
                    help='Log training and validation metrics to mlflow')
#parser.add_argument('--device', type=str, default='cuda')


def main(variant):
    env_name, dataset = variant['env'], variant['dataset']
    data_dir_prefix = variant['data_dir_prefix']
    mode = variant['mode']
    K = variant['K']
    pct_traj = variant['pct_traj']
    model_type = variant['model_type']
    seed = variant['seed']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  #variant.get('device', 'cuda')

    log_to_wandb = variant.get('log_to_wandb', False)
    if log_to_wandb:
        if has_wandb:
            wandb_name = f"{env_name}-{dataset}"
            group_name = model_type + '-' + wandb_name
            wandb.init(
                name=wandb_name,
                group=group_name,
                project='[Decision Mamba] Gym',
                config=variant,
            )
        else:
            print("You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    set_seed(seed)

    #* (1/4) create dataset
    env, max_ep_len, env_targets, scale = get_env_info(env_name, dataset)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if dataset == 'medium-expert':
        dataset_path = data_dir_prefix + f'{env_name}-expert-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        dataset_path = data_dir_prefix + f'{env_name}-medium-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    else:
        dataset_path = data_dir_prefix + f'{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)


    #* (2/4) print dataset info
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print()
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)


    #* (3/4) some settings for training
    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    environment = env_name + "_" + dataset
    save_model_name = model_type + "_" + environment + ".cpt"


    # setup output dir and yaml file
    output_dir = None
    if variant['experiment']:
        exp_name = variant['experiment'] + str(seed)
    else:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_type
            ])
    output_dir = get_outdir(variant['output'] if variant['output'] else './output/train', exp_name)
    args_dir = os.path.join(output_dir, 'args.yaml')
    with open(args_dir, 'w') as f:
        f.write(yaml.safe_dump(variant, default_flow_style=False))


    #***** ** utils ** *****
    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            for _ in range(variant['num_eval_episodes']):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device
                    )
                returns.append(ret)
                lengths.append(length)
            reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
            reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            return {
                f'target_{target}_return_mean': np.mean(returns),
                f'target_{target}_return_std': np.std(returns),
                f'target_{target}_length_mean': np.mean(lengths),
                f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),  # compute the normalized reward, see https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py#L71
            }
        return fn
    #***** ***** ***** *****


    #* (4/4) train model
    model, optimizer, scheduler = get_model_optimizer(variant, state_dim, act_dim, max_ep_len, device)
    print(f"{model_type}: #parameters = {sum(p.numel() for p in model.parameters())}")
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)

    trainer = Trainer(
        model_type=model_type,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=variant['batch_size'],
        get_batch=get_batch,
        loss_fn=loss_fn,
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    n_iter = 0
    try:
        for _iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=_iter+1, print_logs=True)
            if output_dir is not None:
                update_summary(
                    _iter,
                    outputs,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    args_dir=args_dir,
                    write_header=_iter == 0,
                    log_wandb=log_to_wandb and has_wandb,
                    log_mlflow=variant['log_to_mlflow'] and has_mlflow,
                    )
            n_iter += 1
    except KeyboardInterrupt:
        pass

    save_state = {
        'epoch': n_iter+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_state, os.path.join(output_dir, save_model_name))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.log_to_mlflow and has_mlflow:
        mlflow.set_experiment(os.path.basename(args.output))
        with mlflow.start_run(run_name=args.experiment + "_" + str(args.seed)):
            mlflow.log_artifact('./experiment.py')
            mlflow.log_params({k: v for k, v in vars(args).items()})
            main(variant=vars(args))
    else:
        main(variant=vars(args))
