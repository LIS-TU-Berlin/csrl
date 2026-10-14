import math

import numpy as np
import torch
from gymnasium import Env, spaces

from .mujoco_sim import *
from .poly_ref import SecondOrderPolyRef

###############################################################################


class MujocoGym(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    render_mode = "human"
    verbose = 0

    def __init__(self, sim: MujocoSim, cfg, num_scenes=1, terminal_limits=None):
        self.sim = sim
        self.device = self.sim.device

        self.cfg = cfg
        if self.cfg.action_eff == "none":
            self.cfg.action_eff = None
        if self.cfg.goal_feat_points == []:
            self.cfg.goal_feat_points = None
        if self.cfg.obs_points == []:
            self.cfg.obs_points = None
        if self.cfg.obs_pos_offset == []:
            self.obs_pos_offset = None
        else:
            self.obs_pos_offset = torch.as_tensor(self.cfg.obs_pos_offset, dtype=torch.float32, device=self.device)

        self.terminal_limits = torch.as_tensor(terminal_limits, dtype=torch.float32, device=self.device)
        self.num_scenes = num_scenes
        if self.num_scenes > 1:
            self.terminal_limits = self.terminal_limits.reshape(2, num_scenes, -1)
            self.terminal_limits = self.terminal_limits[:, 0, :]
        self.num_worlds = 1
        if self.sim.warp_worlds > 0:
            self.num_worlds = self.sim.warp_worlds
            self.num_scenes *= self.sim.warp_worlds
        self.scene_needs_reset = torch.ones((self.num_scenes), dtype=torch.bool, device=self.device)  # (s)
        self.scene_time = torch.zeros((self.num_scenes), dtype=torch.float32, device=self.device)  # (s)
        self.ctrl_indices = self.sim.ctrl_indices.reshape(self.num_scenes // self.num_worlds, -1)[0]  # (nu)
        self.ctrl_vindices = self.sim.ctrl_vindices.reshape(self.num_scenes // self.num_worlds, -1)[0]  # (nu)
        self.ctrl_current = self.qpos[:, self.ctrl_indices].clone()  # (ns, nu)

        # before evaluating observations and goal features, we grab body_ids for the observation/goal_points as well as potential action_eff control
        if self.cfg.goal_feat_points is not None:
            self.goal_feat_pt_ids = torch.as_tensor(self.sim.get_pt_ids(self.cfg.goal_feat_points), dtype=torch.int32, device=self.device)
            self.goal_feat_pt_ids = self.goal_feat_pt_ids.reshape((len(self.cfg.goal_feat_points), num_scenes)).t()  # (ns, npt)
        if self.cfg.obs_points is not None:
            self.observation_pt_ids = torch.as_tensor(self.sim.get_pt_ids(self.cfg.obs_points), dtype=torch.int32, device=self.device)
            self.observation_pt_ids = self.observation_pt_ids.reshape((len(self.cfg.obs_points), num_scenes)).t()  # (ns, npt)
        if self.cfg.action_eff is not None:
            self.eff_pt_ids = torch.as_tensor(self.sim.get_pt_ids([self.cfg.action_eff]), dtype=torch.int32, device=self.device)
            self.eff_pt_ids = self.eff_pt_ids.reshape(1, num_scenes).transpose()  # (ns, 1)

        # to get the first observation, we need to setup a ctrlRef, get a goal feature, then query an observation
        self.goal_feat = torch.zeros((self.num_scenes, 3 * (self.goal_feat_pt_ids.shape[1] - 1)), dtype=torch.float32, device=self.device)  # the last is the reference!
        self.goal_q = self.qpos.clone()  # just to initialize somehow
        self.observation, self.feat, obs_info = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat, info_string=True)
        self.observation_space = spaces.Box(-2.0, +2.0, shape=tuple(self.observation.shape), dtype=np.float32)  # not possible to use torch directly
        self.single_observation_space = spaces.Box(-2.0, +2.0, shape=(self.observation.shape[1],), dtype=np.float32)

        # define the action space
        self.action_scale = self.cfg.action_scale * math.sqrt(self.cfg.tau_step)
        if self.cfg.action_eff is None:
            self.action_dim = self.sim.model.nu // (self.num_scenes // self.num_worlds)
        else:
            self.action_dim = 3  # 6
            self.q_home = self.qpos[:, self.ctrl_indices_t].detach().cpu().numpy().copy()
        self.action_space = spaces.Box(-1.0, +1.0, shape=(self.num_scenes, self.action_dim), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, +1.0, shape=(self.action_dim,), dtype=np.float32)

        # a counter to loop through provided starts/goals
        self.starts_goals_counter = 0

        # for logging
        H = 100  # horizon of the trace, which is a log of the 0th scene/world; used to write episode files for on-the-fly visualization
        self.trace_qpos = torch.zeros((100, self.qpos.shape[1]), dtype=torch.float32, device=self.device)
        self.trace_reward = torch.zeros((100,), dtype=torch.float32, device=self.device)
        self.trace_succ = torch.zeros((100,), dtype=torch.bool, device=self.device)
        self.trace_truncated = torch.zeros((100,), dtype=torch.bool, device=self.device)
        self.trace_terminated = torch.zeros((100,), dtype=torch.bool, device=self.device)
        # just an accumulating counter -- external writer/reporter needs to reset to zero
        self.total_eps = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.total_steps = 0.0
        self.total_reward = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.total_succ = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.total_terminated = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        print(
            f"-- initialized MjGym with observation {obs_info}, action dim {self.action_dim} (pose_action={self.cfg.action_eff}), tau step {self.cfg.tau_step}, time limit {self.cfg.time_limit}, action scale {self.action_scale}"
        )

    def set_starts_goals(self, starts_q: np.array, starts_r: np.array, goals_q: np.array, goals_feat: np.array):
        assert starts_q.shape[0] == goals_q.shape[0]
        self.starts_goals_counter = 0
        self.starts_q = torch.as_tensor(np.atleast_2d(starts_q), dtype=torch.float32, device=self.device)
        self.starts_v = torch.zeros((starts_q.shape[0], self.sim.model.nv // (self.num_scenes // self.num_worlds)), dtype=torch.float32, device=self.device)
        self.starts_r = torch.as_tensor(np.atleast_2d(starts_r), dtype=torch.float32, device=self.device)
        self.goals_q = torch.as_tensor(np.atleast_2d(goals_q), dtype=torch.float32, device=self.device)
        # self.goals_v = np.zeros((goals_q.shape[0], self.sim.qvel_dim//(self.num_scenes//self.num_worlds)))
        self.goals_feat = torch.as_tensor(np.atleast_2d(goals_feat), dtype=torch.float32, device=self.device)

    def next_starts_goals_counter(self):
        # if self.starts_goals_counter > self.starts_q.shape[0]:
        #     print(f'-- WARNING: reusing start_goals: counter:{self.starts_goals_counter} data:{self.starts_q.shape[0]}')
        i = self.starts_goals_counter % self.starts_q.shape[0]
        self.starts_goals_counter += 1
        return i

    def auto_reset(self):
        assert self.num_scenes > 0
        sel = torch.where(self.scene_needs_reset)[0]
        if sel.numel() == 0:
            return self.observation, {"feature": self.feat}

        # pick num_resets starts/goals indices, via a wrapping counter
        sg_sel = torch.arange(self.starts_goals_counter, self.starts_goals_counter + sel.numel(), device=self.device) % self.starts_q.shape[0]
        self.starts_goals_counter += sel.numel()

        qpos = self.qpos.clone()
        qvel = self.qvel.clone()
        act = self.act.clone()

        # overwrite qpos, qvel, act, self.ctrl_current for the selected envs
        self.goal_feat[sel] = self.goals_feat[sg_sel]
        self.goal_q[sel] = self.goals_q[sg_sel]
        qpos[sel] = self.starts_q[sg_sel]
        qvel[sel] = self.starts_v[sg_sel]
        act[sel] = 0.0
        self.ctrl_current[sel] = self.starts_r[sg_sel][:, self.ctrl_indices]  # mt: inefficient indexing - what is proper?

        # set state, including mj_forward!, which is needed for observations
        self.sim.set_state(qpos, qvel, act)

        # get observations
        # TIM - might be inefficient, as all observations are recomputed instead of only selected.. could change to selected..
        obs_new, feat_new = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat)
        self.observation[sel] = obs_new[sel]
        self.feat[sel] = feat_new[sel]

        self.scene_needs_reset[sel] = False
        self.scene_time[sel] = 0.0

        return self.observation, {"feature": self.feat}

    def reset(self, seed=None, options=None):
        """resets all scenes in the env"""
        if seed is not None:
            super().reset(seed=int(seed))

        self.scene_needs_reset[:] = True

        return self.auto_reset()

    def step(self, action):
        # assert forces a host sync
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        assert isinstance(action, torch.Tensor)
        if action.ndim == 1:
            action = action.view(self.num_scenes, -1)
        assert action.shape[0] == self.num_scenes
        if self.cfg.action_eff is None:
            assert action.shape[1] == self.ctrl_indices.numel()
        else:
            assert action.shape[1] == 3  # 6

        # is a endeffector delta action?
        if self.cfg.action_eff is not None:
            action = self.convert_action_eff(action)

        # set action
        action_delta = self.action_scale * action
        current_vel = self.qvel[:, self.ctrl_vindices]
        poly = SecondOrderPolyRef(self.sim.ctrl_time, self.ctrl_current, current_vel, action_delta, 2.0 * self.cfg.tau_step)
        ctrl_buffer = poly.sample_buffer(self.sim.ctrl_time, self.sim.ctrl_time + self.cfg.tau_step, self.sim.tau_sim)
        assert math.fabs(ctrl_buffer.shape[0] * self.sim.tau_sim - self.cfg.tau_step) < 1e-10, "ctrl_buffer has wrong size"

        # step
        self.sim.rollout(ctrl_buffer)
        self.scene_time += self.cfg.tau_step
        self.ctrl_current = ctrl_buffer[-1]

        # get obs and truncation
        feat_prev = self.feat
        observation_prev = self.observation
        self.observation, self.feat = self.observation_fct(self.qpos, self.qvel, self.ctrl_current, self.goal_feat)
        reward, succ = self.reward_fct(self.observation, self.feat, feat_prev, self.goal_feat)
        terminated = succ.clone()
        truncated = self.scene_time >= self.cfg.time_limit  # terminated and truncated difference is super important
        if self.terminal_limits is not None:
            truncated = torch.logical_or(truncated, self.is_out_of_limits(self.qpos, self.qvel))
            # terminated = torch.logical_or(terminated, self.is_out_of_bound(self.qpos, self.qvel))
        self.scene_needs_reset = torch.logical_or(terminated, truncated)

        # -- fix nan states
        sel = torch.any(torch.isnan(self.observation), dim=1)
        if sel.sum() > 0:
            print(f"\r*** WARNING: step has nans: {sel.sum()} / {sel.shape} (mj step {self.sim.mj_steps}) ***", end="")
            self.observation[sel] = observation_prev[sel]
            self.feat[sel] = feat_prev[sel]
            reward[sel] = 0.0
            succ[sel] = False
            terminated[sel] = True
            truncated[sel] = False
            self.scene_needs_reset[sel] = True
            mjw.reset_data(self.sim.wp_model, self.sim.data, wp.from_torch(sel))

        # trace and log
        self.trace_qpos = torch.roll(self.trace_qpos, -1, 0)
        self.trace_reward = torch.roll(self.trace_reward, -1, 0)
        self.trace_succ = torch.roll(self.trace_succ, -1, 0)
        self.trace_truncated = torch.roll(self.trace_truncated, -1, 0)
        self.trace_terminated = torch.roll(self.trace_terminated, -1, 0)
        self.trace_qpos[-1, :] = self.qpos[0]  # of only the 0th scene!
        self.trace_reward[-1] = reward[0]
        self.trace_succ[-1] = succ[0]
        self.trace_truncated[-1] = truncated[0]
        self.trace_terminated[-1] = terminated[0]
        self.total_eps += self.scene_needs_reset.sum()
        self.total_steps += self.num_scenes
        self.total_reward += reward.sum()
        self.total_succ += succ.sum()
        self.total_terminated += terminated.sum()

        # if self.num_scenes==1: # for stable_baselines to work..
        #    reward = reward.item()
        return self.observation, reward, terminated, truncated, {"feature": self.feat}

    def reward_fct(self, obs, feat, feat_prev, goal_feat):
        # example method for a reward function -- this should be overloaded
        # this example returns a binary 1/0 indicating is_goal
        # BEWARE: the method needs to be vectorized, returning the vector of rewards for all scenes in the env

        dist = torch.linalg.norm(feat - goal_feat, dim=1)
        is_goal = dist <= self.cfg.goal_feat_eps

        # sparse success reward (which should also trigger termination)
        r = is_goal.float()

        # (locally) dense snap reward (a tipi tent at the goal, but zero outside support of 10*eps)
        if self.cfg.reward_dense_radius > 0.0:
            dist_prev = torch.linalg.norm(feat_prev - goal_feat, dim=1)
            scale = self.cfg.reward_dense_radius * self.cfg.goal_feat_eps
            close_prev = (1.0 - dist_prev / scale).clamp(min=0.0)
            close_now = (1.0 - dist / scale).clamp(min=0.0)
            r += 1.0 * (close_now - close_prev)

        if False:  # TESTING! might be stupid: avoids time steps; instead one should reward time steps
            assert self.cfg.obs_nonctrl_q
            nq = self.terminal_limits.shape[1]
            violation = self.limits_violation(obs[:, :nq])
            r -= 0.5 * violation

        return r, is_goal

    def observation_fct(self, qpos, qvel, cref, goal_feat, selected_scenes=None, info_string=False):
        # example method for an observation function -- this should be overloaded
        # this example concatenates:
        # - joint position, velocity, and ctrl_err observations
        # - additional 3D observation points
        # - and the goal feature
        # BEWARE: the method needs to be vectorized, returning the matrix of observations for all scenes in the env

        # -- joint space observations
        if self.cfg.obs_nonctrl_q:
            o_pos = self.cfg.obs_pos_scale * qpos[:, :]
        else:
            o_pos = (
                self.cfg.obs_pos_scale * qpos[:, self.ctrl_indices]
            )  # self.ctrl_indices excludes the object (previously, :-4 which excludes quaternion of the object (assumed last joint!)
        if self.obs_pos_offset is not None:
            o_pos -= self.cfg.obs_pos_scale * self.obs_pos_offset
        o_vel = self.cfg.obs_vel_scale * qvel  # this INCLUDES! all object velocities, including angular
        o_err = self.cfg.obs_referr_scale * (cref - qpos[:, self.ctrl_indices])
        if self.cfg.action_eff is not None:
            sel = selected_scenes if selected_scenes is not None else range(self.num_scenes)
            Jpos, _ = self.sim.get_Jacobian(self.eff_pt_ids[sel])
            nc = len(self.ctrl_indices)
            ns = len(sel)
            Jpos = Jpos.reshape((ns, 3, self.num_scenes, -1))
            Jpos = Jpos[:, :, sel, :]
            Jpos = np.diagonal(Jpos, axis1=0, axis2=2)
            Jpos = np.transpose(Jpos, (2, 0, 1))
            Jpos = Jpos[:, :, :nc]
            o_err = Jpos * o_err[:, None, :]
            o_err = np.sum(o_err, axis=2)
        if info_string:
            info = f"#scenes:{o_pos.shape[0]} qpos:{o_pos.shape[1]} qvel:{o_vel.shape[1]} err:{o_err.shape[1]}"

        obs = torch.hstack((o_pos, o_vel, o_err))

        # -- point observations
        if self.cfg.obs_points is not None:
            o_pts = self.cfg.obs_pos_scale * self.get_pt_feat(self.observation_pt_ids, selected_scenes)
            # o_pts2 = self.cfg.obs_pos_scale * self.get_effpos_observation(self.cfg.obs_points[:-1], selected_scenes)
            obs = torch.hstack((obs, o_pts))
            if info_string:
                info += f" pts:{o_pts.shape[1]}"

        # -- goal feature observation
        if self.cfg.goal_feat_points is not None:
            feat = self.get_pt_feat(self.goal_feat_pt_ids, selected_scenes)
            # feat2 = self.goal_feat_map(qpos) #, qvel) # for debugging
            # print(np.linalg.norm(feat-feat2))
            assert goal_feat.shape == feat.shape
            if info_string:
                info += f" goal_pts:{feat.shape[1]}"
        else:
            raise Exception("deprecated: only point goal features supported right now")
            # feat = self.goal_feat_map(qpos) #, qvel) #AVOID THIS!! too slow
            if info_string:
                info += f" goal_map:{feat.shape[1]}"
        o_goal = self.cfg.obs_pos_scale * (goal_feat - feat)  # we observe the goal RELATIVE to current
        obs = torch.hstack((obs, o_goal))

        # -- final clipping
        obs = torch.clip(obs, -10.0, 10.0)
        if info_string:
            info += f" total:{obs.shape[1]}"
            return obs, feat, info
        return obs, feat

    # needed to support HER
    def apply_delta_goal_feat_to_observation(self, org_obs, org_goal_feat, new_goal_feat):
        if org_obs.ndim == 2 and org_goal_feat.ndim == 1 and new_goal_feat.ndim == 1:
            nf = org_goal_feat.shape[0]
            obs = org_obs.clone()
            obs[:, -nf:] += self.cfg.obs_pos_scale * (new_goal_feat - org_goal_feat)
            return obs
        if org_obs.ndim == 2 and org_goal_feat.ndim == 2 and new_goal_feat.ndim == 2:
            nf = org_goal_feat.shape[1]
            obs = org_obs.clone()
            obs[:, -nf:] += self.cfg.obs_pos_scale * (new_goal_feat - org_goal_feat)
            return obs
        raise Exception("impossible case")

    # needed to support HER
    def get_feat_from_observation(self, obs, goal_feat):
        if obs.ndim == 1:
            assert obs.ndim == 1 and goal_feat.ndim == 1
            nf = goal_feat.shape[0]
            return goal_feat - (obs[-nf:] / self.cfg.obs_pos_scale)
        elif obs.ndim == 2 and goal_feat.ndim == 1:
            nf = goal_feat.shape[0]
            return goal_feat - (obs[:, -nf:] / self.cfg.obs_pos_scale)
        elif obs.ndim == 2 and goal_feat.ndim == 2:
            # assert obs.shape[0]==goal_feat.shape[0]
            nf = goal_feat.shape[1]
            return goal_feat - (obs[:, -nf:] / self.cfg.obs_pos_scale)
        raise Exception("impossible case")

    # helper to compute point observations
    def get_pt_feat(self, pt_ids, selected_scenes=None):
        if selected_scenes is None:
            pts = self.sim.get_pts(pt_ids)
            # the last is the reference!
            pts -= pts[:, -1:, :]
            pts = pts[:, :-1, :]
            return torch.as_tensor(pts.reshape(pts.shape[0], 3 * (pt_ids.shape[1] - 1)), dtype=torch.float32, device=self.device)
        else:
            pts = self.sim.get_pts(pt_ids[selected_scenes])
            # the last is the reference!
            pts -= pts[:, -1:, :]
            pts = pts[:, :-1, :]
            return torch.as_tensor(pts.reshape(len(selected_scenes), 3 * (pt_ids.shape[1] - 1)), dtype=torch.float32, device=self.device)

    def limits_violation(self, qpos):
        violation = torch.zeros((qpos.shape[0]), dtype=torch.float32, device=qpos.device)
        torch.maximum(violation, torch.max(self.terminal_limits[0] - qpos, dim=1)[0], out=violation)
        torch.maximum(violation, torch.max(qpos - self.terminal_limits[1], dim=1)[0], out=violation)

        if self.cfg.debug_limits and violation > self.cfg.limits_margin:
            print("== limit violation debugging:")
            assert self.num_scenes == 1
            j_names = np.array(self.sim.C.getJointNames())
            assert j_names.size == qpos.shape[1]
            vio = (self.terminal_limits[0] - qpos).reshape(-1)
            vio -= self.cfg.limits_margin
            i = torch.where(vio > 0)[0]
            print("   violated lower bound", i, j_names[i], vio[i])
            vio = (qpos - self.terminal_limits[1]).reshape(-1)
            vio -= self.cfg.limits_margin
            i = torch.where(vio > 0)[0]
            print("   violated upper bound", i, j_names[i], vio[i])

        return violation

    def is_out_of_limits(self, qpos, qvel):
        if self.terminal_limits is None or self.cfg.limits_margin < 0.0:
            return torch.zeros((qpos.shape[0]), dtype=torch.bool, device=qpos.device)

        violation = self.limits_violation(qpos)
        return violation - self.cfg.limits_margin > 0.0

    # helper to convert endeffector actions
    def convert_action_eff(self, action, pos_only=True):
        ns = self.num_scenes
        nc = len(self.ctrl_indices)
        pose_action = action
        action = np.zeros((ns, nc))
        # qpos is a torch view post-migration; snapshot to numpy for the per-row math below.
        qpos = self.qpos.detach().cpu().numpy()
        for s in range(ns):
            Jpos, Jang = self.sim.get_Jacobian(self.eff_pt_ids[s])
            if pos_only:
                J = Jpos[0]
                J = J.reshape(3, ns, -1)
            else:
                J = np.vstack((Jpos[0], Jang[0]))
            J = J.reshape(6, ns, -1)
            J = J[:, s, :nc]
            Jinv = J.T @ np.linalg.pinv(J @ J.T + 1e-3 * np.eye(J.shape[0]))
            action[s, :] = Jinv @ pose_action[s, :]
            if self.q_home is not None:
                action[s, :] += 0.1 * (np.eye(nc) - Jinv @ J) @ (self.q_home - qpos[s, :nc])
        return action

    ### minimal example to evaluate a policy

    def rollout(self, pi, return_data=False):
        """helper to play and view a policy"""

        if return_data:
            data = {"state": [], "obs": [], "action": [], "ctrl_cost": [], "next_obs": [], "reward": [], "terminal": []}

        t = 0
        R = 0.0
        while True:
            obs, info = self.auto_reset()
            action = pi(obs, t)  # only for testing we allow for a time-dependent policy
            self.sim.ctrl_costs = 0.0
            next_obs, reward, terminated, truncated, info = self.step(action)
            if return_data:
                data["obs"].append(obs)
                data["action"].append(action)
                data["ctrl_cost"].append(self.sim.ctrl_costs)
                data["reward"].append(np.array([reward]))
                data["next_obs"].append(next_obs)
                data["terminal"].append(np.array([(1 if terminated else 0)], dtype=np.int16))
            R += reward
            t += 1
            if self.verbose > 1:
                print("reward: ", reward)
            if torch.any(torch.logical_or(terminated, truncated)):
                break

        if self.verbose > 0:
            print("total (non-discounted) return:", R)

        if return_data:
            for key, value in data.items():
                data[key] = np.stack(value)
            return data

    @property
    def qpos(self) -> torch.Tensor:
        return self.sim._qpos_t.view(self.num_scenes, -1)

    @property
    def qvel(self) -> torch.Tensor:
        return self.sim._qvel_t.view(self.num_scenes, -1)

    @property
    def act(self) -> torch.Tensor:
        return self.sim._act_t.view(self.num_scenes, -1)  # actuator_force

    def render(self):
        """also part of the env.Gym"""
        self.C.view(False, f"RoboticGym time {self.time} / {self.cfg.time_limit}")
        if self.render_mode == "rgb_array":
            return self.C.view_getRgb()


#    def getattr_depth_check(self, name: str, already_found: bool) -> Optional[str]:
#        return None
#    def step_async(self, actions: np.ndarray) -> None:
#        self.async_actions = actions
#    def step_wait(self):
#        return self.step(self.async_actions)
