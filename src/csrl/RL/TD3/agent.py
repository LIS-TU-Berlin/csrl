import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .buffer import ReplayBuffer

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3_cfg:
    tag = "nil"
    discount = 0.99
    target_tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    bc_weight = 0.5
    value_clip = [0.0, 1.0]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hdim=256):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.leaky_relu(self.l1(state))
        a = F.leaky_relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hdim=256):
        super().__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hdim)
        self.l5 = nn.Linear(hdim, hdim)
        self.l6 = nn.Linear(hdim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(sa))
        q2 = F.leaky_relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.leaky_relu(self.l1(sa))
        q1 = F.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_Agent:
    def __init__(self, env: gym.Env, device, cfg: TD3_cfg):

        self.device = device
        print("== using device:", self.device)
        self.cfg = cfg

        state_dim = env.observation_space.shape[-1]
        action_dim = env.action_space.shape[-1]
        self.max_action = float(np.max(env.action_space.high))

        self.actor = Actor(state_dim, action_dim, self.max_action, hdim=self.cfg.actor_hdim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.learning_rate)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hdim=self.cfg.critic_hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate)
        self.critic_target = copy.deepcopy(self.critic)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, cfg.buffer_size)

        self.clone_buffer = None
        self.training_steps = 0

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def select_action(self, state: torch.Tensor, use_exploration=True):
        with torch.no_grad():
            action = self.actor(state).clamp(-1.0, 1.0)
            if use_exploration:
                action = action + torch.randn_like(action) * self.cfg.exploration_noise
        return action

    def evaluate_critic(self, state: torch.Tensor):
        with torch.no_grad():
            action = self.actor(state).clamp(-1.0, 1.0)
            Q = self.critic.Q1(state, action)
        return Q

    def train(self, writer):
        self.training_steps += 1

        # Sample replay buffer
        _, state, action, next_state, reward, not_done = self.replay_buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.cfg.target_policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.cfg.discount * target_Q
            target_Q = target_Q.clip(*self.cfg.value_clip)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        writer.add("train/Q_loss", critic_loss, 1.0)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.training_steps % self.cfg.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            writer.add("train/pi_value", -actor_loss, 1.0)

            if self.clone_buffer is not None:
                state, action, Return = self.clone_buffer.sample(self.cfg.batch_size)
                BC_loss = F.mse_loss(self.actor(state), action)
                actor_loss = actor_loss + self.bc_weight * BC_loss
                writer.add("train/BC_loss", BC_loss, 1.0)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.cfg.target_tau * param.data + (1 - self.cfg.target_tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.cfg.target_tau * param.data + (1 - self.cfg.target_tau) * target_param.data)
