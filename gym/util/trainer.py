import time
import numpy as np
import torch
import gym

from models.decision_transformer import DecisionTransformer
from models.decision_mamba import DecisionMamba


class Trainer:
    def __init__(
            self,
            model_type,
            model,
            optimizer,
            batch_size,
            get_batch,
            loss_fn,
            scheduler=None,
            eval_fns=None,
            ):
        self.model_type = model_type
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    #** train one iter **
    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()
        self.model.train()
        for i in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

            if i % 1000 == 0:
                print(f'Step {i}')

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/num_of_updates'] = iter_num * num_steps
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    #** train one step **
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        if self.model_type == 'dt':
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg[:,:-1], timesteps, attention_mask=mask
            )
        else:
            action_preds = self.model.forward(states, actions, rtg[:,:-1], timesteps)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]  # no need to care about the future actions
        action_target = action_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


def get_env_info(env_name, dataset):
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [1800, 3600, 7200, 36000, 72000]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [6000, 12000, 24000, 120000, 240000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [2500, 5000, 10000, 50000, 100000]
        scale = 1000.
    #elif env_name == 'antmaze':
        #import d4rl
    #    env = gym.make(f'{env_name}-{dataset}-v2')
    #    max_ep_len = 1000
    #    env_targets = [1.0, 10.0, 1000.0, 100000.0] # successful trajectories have returns of 1, unsuccessful have returns of 0
    #    scale = 1.
    else:
        raise NotImplementedError

    return env, max_ep_len, env_targets, scale


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        device='cuda',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode == 'normal':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def get_model_optimizer(variant, state_dim, act_dim, max_ep_len, device):
    if variant['model_type'] == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=variant['embed_dim'],
            max_length=variant['K'],
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif variant['model_type'] in ["dmamba-min", "dmamba"]:
        model = DecisionMamba(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=variant['embed_dim'],
            max_length=variant['K'],
            max_ep_len=max_ep_len,
            remove_act_embs=variant['remove_act_embs'],
            model_type=variant['model_type'],
            n_layer=variant['n_layer'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            resid_pdrop=variant['dropout'],
            drop_p=variant['dropout'],
            window_size=variant['conv_window_size'],
        )
    else:
        raise NotImplementedError
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    return model, optimizer, scheduler
