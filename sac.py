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
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.net1 = MLP(obs_size+act_size, 1, hidden_size)
        self.net2 = MLP(obs_size+act_size, 1, hidden_size)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)


class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.act_size = act_size
        self.net = MLP(obs_size, act_size*2, hidden_size)

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x[:, :self.act_size], x[:, self.act_size:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()

        # Enforcing action bounds
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def select_action(self, state, device, sample=True):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        mean, log_std = self.forward(state)
        if sample:
            normal = Normal(mean, log_std.exp())
            x = normal.rsample()
        else:
            x = mean
        action = torch.tanh(x)
        return action[0].detach().cpu().numpy()


class SAC:
    def __init__(self,
            device,
            obs_size,
            act_size,
            hidden_size=256,
            gamma=0.99,
            tau=0.005
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self._timestep = 0

        self.critic = Critic(obs_size, act_size, hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.critic_target = Critic(obs_size, act_size, hidden_size).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = Actor(obs_size, act_size, hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.target_entropy = -act_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.replay_buffer = deque(maxlen=1000000)

    def act(self, state, train=True):
        return self.actor.select_action(state, self.device, sample=train)

    def update_parameters(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, k=batch_size)
        state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(self.device) for t in zip(*batch)]

        alpha = self.log_alpha.exp().item()

        # Update critic

        with torch.no_grad():
            next_action, next_action_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            value_next = q_next - alpha * next_action_log_prob
            q_target = reward + not_done * self.gamma * value_next

        q1, q2 = self.critic(state, action)
        q1_loss = 0.5*F.mse_loss(q1, q_target)
        q2_loss = 0.5*F.mse_loss(q2, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)

        # Update actor

        action_new, action_new_log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha*action_new_log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha

        alpha_loss = -(self.log_alpha.exp() * (action_new_log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
