import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    """ MLP with dense connections """
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers=3):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        hidden_size_aug = hidden_size + input_size
        self.linear_in = nn.Linear(input_size, hidden_size)
        hidden_layers = []
        for i in range(self.num_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_size_aug, hidden_size))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        x = F.relu(self.linear_in(inp))
        for i in range(self.num_hidden_layers):
            x = torch.cat([x, inp], dim=1)
            x = F.relu(self.hidden_layers[i](x))
        return self.linear_out(x)


class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self, obs_size, act_size, hidden_size, num_nets):
        super().__init__()
        self.nets = nn.ModuleList([
            MLP(obs_size+act_size, 1, hidden_size)
            for _ in range(num_nets)
        ])

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return [net(state_action) for net in self.nets]


class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size, max_action):
        super().__init__()
        self.net = MLP(obs_size, act_size, hidden_size)
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        action = torch.tanh(x) * self.max_action
        return action

    def act(self, state, device, noise=0):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = self.forward(state)
        return action[0].detach().cpu().numpy()


class REDQ:
    def __init__(self,
        device,
        obs_size,
        act_size,
        max_action=1,
        hidden_size=256,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        exploration_noise=0.1,
        critic_num_nets=10,
    ):
        self.device = device
        self.act_size = act_size
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self.critic_num_nets = critic_num_nets
        self._timestep = 0

        self.critic = Critic(obs_size, act_size, hidden_size, critic_num_nets).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.critic_target = Critic(obs_size, act_size, hidden_size, critic_num_nets).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.actor = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.actor_target = Actor(obs_size, act_size, hidden_size, max_action).to(device)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        self.replay_buffer = []

    def act(self, state, train=True):
        action = self.actor.act(state, self.device)
        if train:
            action = (
                action + np.random.normal(0, self.exploration_noise, size=self.act_size)
            ).clip(-self.max_action, self.max_action)
        return action

    def update_parameters(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, k=batch_size)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(self.device) for t in zip(*batch)]

        # Update critic

        with torch.no_grad():
            noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            qs_next = self.critic_target(next_state, next_action)
            random_idxs = np.random.permutation(self.critic_num_nets)
            q1_next = qs_next[random_idxs[0]]
            q2_next = qs_next[random_idxs[1]]
            q_next = torch.min(q1_next, q2_next)
            q_target = reward + not_done * self.gamma * q_next

        qs = self.critic(state, action)
        critic_loss = sum([F.mse_loss(q, q_target) for q in qs])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor

        if self._timestep % self.policy_freq == 0:
            action_new = self.actor(state)
            qs_new = self.critic(state, action_new)
            q_new = torch.stack(qs_new, 0).mean(0)
            actor_loss = -q_new.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)
