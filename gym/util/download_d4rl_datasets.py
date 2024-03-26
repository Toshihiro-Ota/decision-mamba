import argparse
import collections
import pickle

import numpy as np
import gym
import d4rl


parser = argparse.ArgumentParser()
parser.add_argument('--suite', type=str, choices=['locomotion', 'antmaze'], default='locomotion',
					help='Which suite of environments to download (default: "locomotion")')
parser.add_argument('--data-dir', metavar='DIR', default='./data/',
                    help='Path to dataset (default: "./data/", must include the last slash "/")')


datasets = []

def main():
	args = parser.parse_args()

	if args.suite == "locomotion":
		env_names = ['halfcheetah', 'hopper', 'walker2d']
		dataset_types = ['medium', 'medium-replay', 'expert']
		names = [f'{env_name}-{dataset_type}-v2' for env_name in env_names for dataset_type in dataset_types]
		keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
	elif args.suite == "antmaze":
		names = ['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2',
				'antmaze-medium-play-v2', 'antmaze-medium-diverse-v2',
				'antmaze-large-play-v2', 'antmaze-large-diverse-v2']
		keys = ['observations', 'actions', 'rewards', 'terminals']

	for name in names:
		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = 'timeouts' in dataset

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				raise RuntimeError('All datasets should have timeouts')
				# final_timestep = (episode_step == 1000-1)
			for k in keys:
				data_[k].append(dataset[k][i])
			end_of_episode = done_bool or final_timestep
			if end_of_episode:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(args.data_dir + f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)


if __name__ == '__main__':
	main()
