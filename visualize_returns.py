import argparse
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualize returns from RealAnt')
parser.add_argument('--exp_folder', type=str)
args = parser.parse_args()

list_of_files = glob.glob(f'{args.exp_folder}/episode_*.pickle')

returns = {}
for filename in list_of_files:
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 2:
        transitions, info = data
    else:
        transitions, info = data, None

    obses, actions, rewards, _, _ = zip(*transitions)
 
    episode = int(filename.split('/')[-1].split('.')[0].split('_')[-1])
    returns[episode] = sum([r[0] for r in rewards])

sorted_episodes = sorted(returns.keys())
sorted_returns = [returns[ep] for ep in sorted_episodes]

plt.figure()
plt.plot(sorted_episodes, sorted_returns)
plt.xlim(sorted_episodes[0], sorted_episodes[-1])
plt.ylabel('Returns')
plt.xlabel('Episodes')
plt.title(f'Run {args.exp_folder}')
plt.savefig(f'{args.exp_folder}/returns.png')
