import argparse
import random
import numpy as np
import torch

import gym
import realant_sim

from td3 import TD3
from sac import SAC

def rollout(agent, env, train=False, random=False):
    state = env.reset()
    episode_step, episode_return = 0, 0
    done = False
    while not done:
        if random:
            action = env.action_space.sample()
        else:
            action = agent.act(state, train=train)

        next_state, reward, done, info = env.step(action)
        episode_return += reward

        if train:
            not_done = 1.0 if (episode_step+1) == env._max_episode_steps else float(not done)
            agent.replay_buffer.append([state, action, [reward], next_state, [not_done]])
            agent._timestep += 1

        state = next_state
        episode_step += 1

    if train and not random:
        for _ in range(episode_step):
            agent.update_parameters()

    return episode_return

def evaluate(agent, env, n_episodes=10):
    returns = [rollout(agent, env, train=False, random=False) for _ in range(n_episodes)]
    return np.mean(returns)

def train(agent, env, n_episodes=1000, n_random_episodes=10):
    for episode in range(n_episodes):
        train_return = rollout(agent, env, train=True, random=episode<n_random_episodes)
        print(f'Episode {episode}. Return {train_return}')

        if (episode+1) % 10 == 0:
            eval_return = evaluate(agent, env)
            print(f'Eval Reward {eval_return}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="td3")  # td3 or sac
    parser.add_argument("--env", default="mujoco") # mujoco or pybullet
    parser.add_argument("--task", default="walk")  # sleep or turn or walk

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--latency", default=2, type=int)
    parser.add_argument("--xyz_noise_std", default=0.01, type=int)
    parser.add_argument("--rpy_noise_std", default=0.01, type=int)
    parser.add_argument("--min_obs_stack", default=4, type=int)

    args = parser.parse_args()

    if args.env == 'mujoco':
        env =  gym.make(
            'RealAntMujoco-v0',
            task=args.task,
            latency=args.latency,
            xyz_noise_std=args.xyz_noise_std,
            rpy_noise_std=args.rpy_noise_std,
            min_obs_stack=args.min_obs_stack,
        )
    elif args.env == 'pybullet':
        env = gym.make('RealAntBullet-v0', task=args.task)
    else:
        raise Exception('Unknown env')

    obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.agent == 'td3':
        agent = TD3(device, obs_size, act_size)
    elif args.agent == 'sac':
        agent = SAC(device, obs_size, act_size)
    else:
        raise Exception('Unknown agent')

    train(agent, env, n_episodes=1000, n_random_episodes=10)
