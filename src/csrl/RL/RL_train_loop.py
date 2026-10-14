import os
import time

import gymnasium as gym
import numpy as np
import psutil
import torch
from robotic.tools.h5_helper import H5Writer

from .writer import Writer

process = psutil.Process()


def proc_info():
    ci = process.cpu_times()
    mi = process.memory_info()
    gpu = os.getenv("CUDA_VISIBLE_DEVICES")
    GB = 1024 * 1024 * 1024
    return f"user: {ci.user / 3600:.2f}h, rsize: {mi.rss / GB:.2f}GB, vsize: {mi.vms / GB:.2f}GB, gpu: {gpu}"


class RunningFilter:
    def __init__(self, n, gamma=1.0, device=None):
        self.device = device
        self.gamma = gamma
        self.n = torch.zeros((), device=device)
        self.m1 = torch.zeros((n,), device=device)
        self.m2 = torch.zeros((n,), device=device)

    def update(self, x):
        assert x.ndim == 2
        self.n += x.shape[0]
        self.m1 += torch.sum(x, dim=0)
        self.m2 += torch.sum(x * x, dim=0)

    def apply(self, x):
        return x / ((self.m2 / self.n).sqrt() + 0.1)

    def print(self):
        s2 = (self.m2 / self.n).sqrt().detach().cpu().numpy()
        n = float(self.n.item())
        return f"n: {n} s2: {s2}"


def randint(upper_int_tensor):
    rnd_ints = torch.randint(0, torch.iinfo(torch.int32).max, size=upper_int_tensor.shape, device=upper_int_tensor.device)
    return rnd_ints % upper_int_tensor


def rewrite_goal_feature(env, S, N, R, C, org_goal_feat, new_goal_feat):
    S = env.apply_delta_goal_feat_to_observation(S, org_goal_feat, new_goal_feat)
    N = env.apply_delta_goal_feat_to_observation(N, org_goal_feat, new_goal_feat)
    S_feat = env.get_feat_from_observation(S, new_goal_feat)
    N_feat = env.get_feat_from_observation(N, new_goal_feat)
    R, is_goal = env.reward_fct(obs=N, feat=N_feat, feat_prev=S_feat, goal_feat=new_goal_feat)
    C = (~is_goal).float()
    return S, N, R, C


class RL_train_loop:
    def __init__(self, env: gym.Env, agent, device, cfg):
        self.env = env

        # state_dim = env.observation_space.shape[-1]
        # self.action_dim = env.action_space.shape[-1]
        # max_action = float(np.max(env.action_space.high))
        # self.agent = TD7_Agent(state_dim, self.action_dim, max_action, offline=False, hp=cfg)
        self.agent = agent
        self.device = device
        self.cfg = cfg

        # self.state_filter = RunningFilter(state_dim, device=self.device)

        self.num_envs = self.env.num_scenes
        if self.num_envs > 1:
            assert self.num_envs == env.observation_space.shape[0]

        self.tot_time = 1e-6
        self.reset_time = 0.0
        self.step_time = 0.0
        self.her_time = 0.0
        self.train_time = 0.0
        self.Qeval_time = 0.0
        self.writer = Writer(f"{cfg.tag}")

        self.t = 0
        self.next_report_t = 0
        self.step_data = 0
        self.HER_data = 0
        self.buffer_prev_index = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)

        # replay buffer traces to enable HER:
        #   rolling trace indices (each row contains the replay buffer indices of consecutive steps in a single env; #rows = #envs; #cols=20 is fixed ad hoc)
        #   ep_steps: the number of consecutive steps since the last reset in each env -> HER should not go further back
        self.rolling_trace_indices = torch.zeros((self.num_envs, 40), dtype=torch.int32, device=self.device)
        self.ep_finished = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.ep_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)

        # stats for writer
        self.act_hist = torch.zeros((20,), dtype=torch.float32, device=self.device)
        self.obs_hist = torch.zeros((20,), dtype=torch.float32, device=self.device)
        self.hist_count = 0.0
        self.ep_reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.ep_Q = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.best_reward = -1e10

    def train(self, train_steps, reset_num_timesteps=None, log_interval=None, tb_log_name=None):

        t_stop = self.t + train_steps
        while self.t < t_stop:
            # -- print info, log model
            self.report_callback(self.env)
            self.tot_time -= time.time()

            # -- clear writer/logging states
            sel = torch.where(self.ep_finished)[0]
            if sel.numel() > 0:
                self.ep_steps[sel] = 0
                self.ep_reward[sel] = 0.0
                self.ep_Q[sel] = 0.0

            # -- multiple scenes: auto_reset
            self.reset_time -= time.time()
            state, info = self.env.auto_reset()
            self.reset_time += time.time()

            # self.state_filter.update(state)
            # if self.cfg.obs_normalize:
            #     state = self.state_filter.apply(state)

            # -- select action
            if self.agent.replay_buffer.size >= self.cfg.datasize_before_execution:
                action = self.agent.select_action(state, use_exploration=True)
            else:
                # action = torch.rand(state.shape[0], self.action_dim, device=self.device) * 2. - 1.
                action = (0.5 * torch.randn(state.shape[0], self.env.action_space.shape[-1], device=self.device)).clamp(-1.0, +1.0)
                # action = self.env.action_space.sample().reshape(state.shape[0], -1) #gymnasium Box() samples in numpy -> directly sample with torch
                # action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

            # -- single step
            self.step_time -= time.time()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.ep_finished = torch.logical_or(terminated, truncated)
            self.ep_steps += 1
            self.ep_reward += reward
            self.step_time += time.time()

            # if self.cfg.obs_normalize:
            #     next_state = self.state_filter.apply(next_state)

            # -- add to replay buffer, and memorize buffer indices as self.buffer_prev_index
            buffer_index = self.agent.replay_buffer.add(state, action, next_state, reward, terminated, self.buffer_prev_index)
            self.step_data += state.shape[0]
            self.rolling_trace_indices = torch.roll(self.rolling_trace_indices, 1, 1)
            self.rolling_trace_indices[:, 0] = buffer_index

            # -- add HER data to replay buffer
            self.her_time -= time.time()
            if self.cfg.HER_ratio > 0.0:
                # self.HER_old(self.ep_finished, buffer_index)
                self.HER_new(next_state)
            self.her_time += time.time()

            # memorize buffer indices as buffer_prev_index (needs to be done AFTER HER_old)
            self.buffer_prev_index = buffer_index.clone()
            self.buffer_prev_index[self.ep_finished] = -1

            # add to histograms
            self.act_hist += torch.histc(action.view(-1).float(), bins=20, min=-1.0, max=1.0) / action.numel()
            self.obs_hist += torch.histc(state.view(-1).float(), bins=20, min=-2.0, max=2.0) / state.numel()
            self.hist_count += 1.0

            # -- train
            self.train_time -= time.time()
            if self.agent.replay_buffer.size >= self.cfg.datasize_before_training:
                for _ in range(self.cfg.train_steps):
                    self.agent.train(self.writer)
            self.train_time += time.time()

            # -- if truncated, evaluate Q function (takes quite some time...)
            # self.Qeval_time -= time.time()
            # sel = torch.where(truncated)[0]
            # if sel.numel()>0:
            #     self.ep_Q[sel] = self.agent.evaluate_critic(next_state[sel])
            # self.Qeval_time += time.time()

            # -- increment step
            self.t += self.cfg.train_steps
            self.tot_time += time.time()

        self.report_callback(self.env)

    def HER_new(self, next_state):
        ## options to select environments: steps>2, or finished
        sel = torch.where(self.ep_steps > 2)[0]
        # sel = torch.where(self.ep_finished)[0]

        if sel.numel() > 0:
            if self.HER_data < self.step_data * self.cfg.HER_ratio:  ## option: replace by 'while' if you want HER ratio>1
                org_goal_feat = self.env.goal_feat[sel]

                ## options to select goal: last (passed as next_state), or random in episode
                N = next_state[sel]
                # rnd_goal_steps = randint(self.ep_steps[sel].clamp(min=None, max=self.rolling_trace_indices.shape[1]))
                # rnd_goal_indices = self.rolling_trace_indices[sel, rnd_goal_steps] #crazy indexing semantics
                # S,A,N,R,C = self.agent.replay_buffer.get_sub(rnd_goal_indices)

                new_goal_feat = self.env.get_feat_from_observation(N, org_goal_feat)
                # print(np.linalg.norm(new_goal_feat - self.env.feat[sel]))  #if we take the next_state goal feature, that should be the same as the 'current'(=next state) state feature

                for _ in range(self.cfg.HER_samples):
                    rnd_rewrite_steps = randint(self.ep_steps[sel].clamp(min=None, max=self.rolling_trace_indices.shape[1]))
                    rnd_rewrite_indices = self.rolling_trace_indices[sel, rnd_rewrite_steps]  # crazy indexing semantics
                    S, A, N, R, C = self.agent.replay_buffer.get_sub(rnd_rewrite_indices)

                    S, N, R, C = rewrite_goal_feature(self.env, S, N, R, C, org_goal_feat, new_goal_feat)
                    self.agent.replay_buffer.add(S, A, N, R, 1.0 - C)
                    self.HER_data += S.shape[0]

    def HER_old(self, ep_finished, buffer_index):
        # add HER data to replay buffer
        if self.cfg.HER_ratio > 0.0 and self.HER_data < self.step_data * self.cfg.HER_ratio:
            ep_finished_cpu = ep_finished.cpu().numpy()
            for i in range(self.num_envs):
                if ep_finished_cpu[i]:  # and not terminated[i]:
                    ep_indices = self.agent.replay_buffer.get_episode_trace(buffer_index[i])
                    if ep_indices.numel() > 3:
                        S, A, N, R, C = self.agent.replay_buffer.get_sub(ep_indices)  # state, action, next state, reward, continue(=not-done)
                        org_goal_feat = self.env.goal_feat[i]
                        new_goal_index = np.random.randint(ep_indices.numel())
                        new_goal_feat = self.env.get_feat_from_observation(N[new_goal_index], org_goal_feat)
                        # assert np.linalg.norm(new_goal_feat - info['feature'][i]) < 1e-10

                        # print('episode len:', len(ep_indices), 'episode end feature:', new_goal_feat, 'org rewards', R.reshape(-1), C.reshape(-1))
                        S, N, R, C = rewrite_goal_feature(self.env, S, N, R, C, org_goal_feat, new_goal_feat)
                        # print('new rewards:', R.reshape(-1), C.reshape(-1))
                        # assert not C[new_goal_index], 'the selected state of a rewritten episode should be a goal!'
                        self.agent.replay_buffer.add(S, A, N, R, 1.0 - C)
                        self.HER_data += S.shape[0]

    def predict(self, state, deterministic=True):
        # state = torch.as_tensor(np.atleast_2d(state), dtype=torch.float32, device=self.agent.device)
        assert isinstance(state, torch.Tensor)
        action = self.agent.select_action(state, use_exploration=not deterministic)
        return action, None  # following sb conventions

    def write_trace(self, env):
        filename = f"data/tmp/{self.writer.tag}_trace.h5"
        h5 = H5Writer(filename)
        info = {"tag": self.writer.tag, "t": self.t}
        h5.write_dict("info", info)
        h5.write("qpos", env.trace_qpos.numpy(force=True), dtype="float32")
        h5.write("reward", env.trace_reward.numpy(force=True), dtype="float32")
        h5.write("succ", env.trace_succ.numpy(force=True), dtype="bool")
        h5.write("truncated", env.trace_truncated.numpy(force=True), dtype="bool")
        h5.write("terminated", env.trace_terminated.numpy(force=True), dtype="bool")

    def write_replay_buffer(self):
        filename = f"data/tmp/{self.writer.tag}_buffer.h5"
        h5 = H5Writer(filename)
        size = self.agent.replay_buffer.size
        manifest = {
            "tag": self.writer.tag,
            "t": self.t,
            "buffer_size": size,
            "buffer_max_size": self.agent.replay_buffer.max_size,
            "current_ptr": self.agent.replay_buffer.ptr,
            "HER_data": self.HER_data,
            "step_data": self.step_data,
        }
        h5.write_dict("manifest", manifest)
        h5.write("state", self.agent.replay_buffer.state[:size].numpy(force=True), dtype="float32")
        h5.write("action", self.agent.replay_buffer.action[:size].numpy(force=True), dtype="float32")
        h5.write("next_state", self.agent.replay_buffer.next_state[:size].numpy(force=True), dtype="float32")
        h5.write("reward", self.agent.replay_buffer.reward[:size].numpy(force=True), dtype="float32")
        h5.write("cont", self.agent.replay_buffer.not_done[:size].numpy(force=True), dtype="bool")
        h5.write("prev_index", self.agent.replay_buffer.prev_index[:size].numpy(force=True), dtype="int32")

    def report_callback(self, env):
        if self.t >= self.next_report_t:
            self.next_report_t += self.cfg.eval_freq

            # -- get log scalars from env
            eps, steps, reward, succ, terminated = (
                env.total_eps.item(),
                env.total_steps,
                env.total_reward.item(),
                env.total_succ.item(),
                env.total_terminated.item(),
            )
            env.total_eps, env.total_steps, env.total_reward, env.total_succ, env.total_terminated = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

            # -- write policy, replay_buffer, and trace into files
            # avg_reward = self.writer.get_value('ep/reward')
            avg_reward = 0
            if eps > 0.0:
                avg_reward = reward / eps
                if avg_reward > self.best_reward:
                    self.agent.save(filename=f"data/tmp/{self.writer.tag}.torch")
                    self.best_reward = avg_reward
            # write replay buffer
            if self.cfg.write_replay_buffer:
                self.write_replay_buffer()
            # write some episodes
            if self.cfg.write_trace:
                self.write_trace(env)

            print("---------------------------------------")
            print(f"Report after {self.t} training steps, tag: {self.cfg.tag}")
            time_passed = self.tot_time
            print(f"Avg reward since last report: {avg_reward} (best: {self.best_reward})")
            print("Total resources:", proc_info())
            print(
                f"Total learning time: {time_passed / 3600.0:.2f}h,",
                f"reset: {self.reset_time / time_passed * 100.0:.1f}%,",
                f"step: {self.step_time / time_passed * 100.0:.1f}%,",
                f"her: {self.her_time / time_passed * 100.0:.1f}%,",
                f"train: {self.train_time / time_passed * 100.0:.1f}%,",
                f"Qeval: {self.Qeval_time / time_passed * 100.0:.1f}%",
            )

            print(
                f"ReplayBuffer size: {self.agent.replay_buffer.size}, step data: {self.step_data}, HER data: {self.HER_data}, HER ratio: {(self.HER_data / self.step_data if self.step_data > 0 else 0) * 100.0:.1f}%"
            )
            if self.hist_count > 0:
                np.set_printoptions(linewidth=np.inf)
                act = self.act_hist.detach().cpu().numpy()
                obs = self.obs_hist.detach().cpu().numpy()
                print(f"action hist [-1.,1.]: {(1000.0 * act / self.hist_count).round().astype(np.int16)}")
                print(f"observ hist [-2.,2.]: {(1000.0 * obs / self.hist_count).round().astype(np.int16)}")
                self.act_hist.zero_()
                self.obs_hist.zero_()
                self.hist_count = 0.0
                # print(f'running state filter: {self.state_filter.print()}')

            if self.writer is not None:
                # if eps>0:
                #     # self.writer.writer.add_scalar("ep/reward", reward/eps, self.t)
                #     self.writer.writer.add_scalar("ep/succ", succ/eps, self.t)
                #     # self.writer.writer.add_scalar("ep/steps", steps/eps, self.t)
                self.writer.add("ep/succ", succ, eps)
                self.writer.add("ep/reward", reward, eps)
                self.writer.add("ep/steps", steps, eps)
                self.writer.add("ep/terminated", terminated, eps)

                self.writer.write(self.t, verbose=1)

            print("---------------------------------------")
