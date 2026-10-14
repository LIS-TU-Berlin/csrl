import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .buffer import *


def _unwrap(module):
    # Fix to load weights into a compiled model
    return getattr(module, "_orig_mod", module)


def param_norms(model):
    param_norm = 0.0
    grad_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm += torch.square(p.data).sum()
        grad_norm += torch.square(p.grad).sum()
    return param_norm**0.5, grad_norm**0.5


@dataclass
class TD7_Config:
    tag: str = "nil"
    # Generic
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.95
    target_update_rate: int = 250
    exploration_noise: float = 0.1
    value_clip = [0.0, 1.0]
    obs_normalize = False

    # TD3
    target_policy_noise: float = 0.1
    noise_clip: float = 0.2
    policy_freq: int = 2

    # LAP
    alpha: float = 0.4
    min_priority: float = 1

    # TD3+BC
    lmbda: float = 0.1

    # HER
    HER_ratio: float = 0.0
    HER_samples: int = 1

    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4
    reset_weight: float = 0.9

    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256

    # Critic Model
    critic_hdim: int = 256

    # Actor Model
    actor_hdim: int = 256

    # Compilation
    compile: bool = False


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super().__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super().__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super().__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class TD7_Agent:
    def __init__(self, state_dim, action_dim, max_action, device, offline=False, cfg=TD7_Config()):
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)

        self.device = device
        print("== using device:", self.device)
        self.cfg = cfg

        self.actor = Actor(state_dim, action_dim, cfg.zs_dim, cfg.actor_hdim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, cfg.zs_dim, cfg.critic_hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, cfg.zs_dim, cfg.enc_hdim).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=cfg.learning_rate)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        if cfg.compile:
            # Compile only the modules
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            self.encoder.zs = torch.compile(self.encoder.zs)
            self.encoder.zsa = torch.compile(self.encoder.zsa)
            self.fixed_encoder.zs = torch.compile(self.fixed_encoder.zs)
            self.fixed_encoder.zsa = torch.compile(self.fixed_encoder.zsa)

        self.replay_buffer = LAP(
            state_dim,
            action_dim,
            self.device,  # , 'cpu', #critical!
            cfg.buffer_size,
            # hp.batch_size,
            max_action,
            normalize_actions=True,
            prioritized=True,
        )

        self.max_action = max_action
        self.offline = offline

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = torch.tensor(-1e8, dtype=torch.float32, device=self.device)
        self.min = torch.tensor(1e8, dtype=torch.float32, device=self.device)
        self.max_target = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.min_target = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def save(self, filename):
        torch.save(
            {
                "actor": _unwrap(self.actor).state_dict(),
                "critic": _unwrap(self.critic).state_dict(),
                "fixed_encoder": self.fixed_encoder.state_dict(),
            },
            filename,
        )

    def load(self, filename, device="cpu"):
        models = torch.load(filename, weights_only=True, map_location=torch.device(device))
        _unwrap(self.actor).load_state_dict(models["actor"])
        _unwrap(self.critic).load_state_dict(models["critic"])
        self.fixed_encoder.load_state_dict(models["fixed_encoder"])

        self.actor_target.load_state_dict(_unwrap(self.actor).state_dict())
        self.critic_target.load_state_dict(_unwrap(self.critic).state_dict())
        self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
        self.encoder.load_state_dict(self.fixed_encoder.state_dict())

    def select_action(self, state: torch.Tensor, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                action = action + torch.randn_like(action) * self.cfg.exploration_noise

            return action.clamp(-1, 1) * self.max_action

    def predict(self, state, deterministic=True):
        assert isinstance(state, torch.Tensor)
        return self.select_action(state, use_exploration=not deterministic), None  # following sb conventions

    def evaluate_critic(self, state: torch.Tensor):
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            action = self.actor(state, zs).clamp(-1, 1)
            zsa = self.fixed_encoder.zsa(zs, action)
            Q = self.critic(state, action, zsa, zs).mean(axis=1)
        return Q

    def train(self, writer, clone_buffer=None):

        indices, state, action, next_state, reward, not_done = self.replay_buffer.sample(self.cfg.batch_size)

        self.train_step(indices, state, action, next_state, reward, not_done, writer, clone_buffer)

    def train_step(self, indices, state, action, next_state, reward, not_done, writer, clone_buffer):
        self.training_steps += 1

        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        writer.add("train/enc_loss", encoder_loss, 1.0)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        pn, gn = param_norms(self.encoder)
        writer.add("train/enc_norm", pn, 1.0)
        writer.add("train/enc_grad_norm", gn, 1.0)

        #########################
        # Update Critic
        #########################
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.cfg.target_policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1, 1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.cfg.discount * Q_target.clamp(self.min_target, self.max_target)

            torch.maximum(self.max, Q_target.max(), out=self.max)
            torch.minimum(self.min, Q_target.min(), out=self.min)

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

            Q_target = Q_target.clip(*self.cfg.value_clip)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)

        writer.add("train/Q_loss", critic_loss, 1.0)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pn, gn = param_norms(self.critic)
        writer.add("train/Q_norm", pn, 1.0)
        writer.add("train/Q_grad_norm", gn, 1.0)

        #########################
        # Update LAP
        #########################
        priority = td_loss.max(1)[0].clamp(min=self.cfg.min_priority).pow(self.cfg.alpha)
        self.replay_buffer.update_priority(priority, indices)

        #########################
        # Update Actor
        #########################
        if self.training_steps % self.cfg.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)

            writer.add("train/pi_value", Q.mean(), 1.0)

            actor_loss = -Q.mean()
            if self.offline:
                actor_loss = actor_loss + self.cfg.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)

            if clone_buffer is not None:
                c_state, c_action = clone_buffer.sample(self.cfg.batch_size)
                c_fixed_zs = self.fixed_encoder.zs(c_state)
                bc_loss = F.mse_loss(self.actor(c_state, c_fixed_zs), c_action)
                actor_loss = actor_loss + self.cfg.lmbda * bc_loss

                writer.add("train/BC_loss", bc_loss, 1.0)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            pn, gn = param_norms(self.actor)
            writer.add("train/pi_norm", pn, 1.0)
            writer.add("train/pi_grad_norm", gn, 1.0)

        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.cfg.target_update_rate == 0:
            self.actor_target.load_state_dict(_unwrap(self.actor).state_dict())
            self.critic_target.load_state_dict(_unwrap(self.critic).state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()

    # Batch training
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.cfg.steps_before_checkpointing:
                self.best_min_return *= self.cfg.reset_weight
                self.max_eps_before_update = self.cfg.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8
