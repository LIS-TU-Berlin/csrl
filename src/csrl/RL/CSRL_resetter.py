import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import robotic as ry
import torch
from robotic.tools import h5_helper, video_gen
from scipy.spatial import KDTree

from csrl.envs.mujoco_gym import MujocoGym

from .TD7.buffer import CloneBuffer

process = psutil.Process()


def proc_mem():
    mi = process.memory_info()
    GB = 1024 * 1024 * 1024
    return mi.vms / GB


class H5MatchFile:
    def __init__(self, h5_filename):
        h5 = h5_helper.H5Reader(h5_filename)
        self.__dict__ = h5.read_all()  # MAGIC!


class ConfigsFile:
    manifest: dict
    supports: list
    config: dict
    q: np.array
    contacts: np.array
    forces: np.array

    def __init__(self, h5_filename):
        h5 = h5_helper.H5Reader(h5_filename)
        self.__dict__ = h5.read_all()  # MAGIC!
        if "potential_supports" in self.manifest:
            self.supports = self.manifest["potential_supports"]
        else:
            self.supports_bot = self.manifest["supports_bot"]
            self.supports = self.manifest["supports_env"]
        print(f"-- loaded config file, mem: {(self.q.nbytes) / (1024 * 1024 * 1024):.2f}GB")


class EpisodesFile(H5MatchFile):
    manifest: dict
    config: dict
    tau_step: float
    task: np.array
    start: np.array
    goal: np.array
    final_cost: np.array
    state: np.array
    action: np.array


class TrajectoryData:
    goal: np.array
    state: np.array
    observation: np.array
    action: np.array
    time_to_go: np.array


class TrajCloneBuffer:
    observation: np.array
    action: np.array
    Return: np.array
    Gamma: np.array
    next_observation: np.array


class CSRL_Resetter:
    def __init__(self, C: ry.Config, goal_feat_map, interp_embedding, cfg):  # creator: Workspace):
        # self.creator = creator
        self.C = C
        self.goal_feat_map = goal_feat_map
        self.interp_embedding = interp_embedding
        self.cfg = cfg

    def update_environment(self, env: MujocoGym, t, td7):
        # assert self.cfg.start_goal_mode!='CC' or self.cfg.T_curr==0, 'can not have a curriculum for direct start-goal sampling'
        assert self.cfg.T_curr > 0, "convention changed: use mode without c instead of T_curr=0"

        # phase
        alpha = (t + self.cfg.T_block) / (self.cfg.T_curr + 1)
        alpha = min(alpha, 1.0)
        env.cfg.time_limit = 2.0 * alpha * self.cfg.nominal_time_limit

        # sample new start/goals for this phase
        n = env.starts_goals_counter
        if n == 0:
            n = (
                self.cfg.T_block * env.num_scenes / (0.1 * self.cfg.nominal_time_limit / env.cfg.tau_step)
            )  # heuristic: enough goals if episodes would last only 20% of nominal time limit
        else:
            n *= 2  # heuristic: twice as many goals as needed in the previous block
        n = int(min(n, 1e5))
        starts, starts_r, goals, goals_feat = self.getStartsGoals(n=n, alpha=alpha, mode=self.cfg.start_goal_mode)
        env.set_starts_goals(starts, starts_r, goals, goals_feat)

        # enable BC?
        if t < self.cfg.T_cloning:
            traj_clone_buffer = self.get_TrajCloneBuffer(alpha)
            td7.clone_buffer = CloneBuffer(traj_clone_buffer.observation, traj_clone_buffer.action)
            cloneInfo = td7.clone_buffer.state.shape[0]
        else:
            td7.clone_buffer = None
            cloneInfo = 0

        print(f"=== t: {t}, alpha: {alpha:.2f}, mode: {self.cfg.start_goal_mode}, starts/goals: {env.starts_q.shape[0]}, clone: {cloneInfo}, time_limit: {env.cfg.time_limit:.2f}")

    def getStartsGoals(self, n, alpha, mode):
        assert alpha > 0.0 and alpha <= 1.0
        goals_feat = None
        starts_r = np.empty(())

        if mode[-1] == "c":  # with curriculum
            mode = mode[:-1]
        elif mode[:2] != "rv":
            alpha = 1.0  # without curriculum

        # if mode=='direct':
        #     return self.getStartsGoals_fromRandomTasks(n, alpha=None)

        if mode == "CC":
            assert alpha == 1.0, "CC does not support curriculum"
            starts_q, starts_z, starts_r = self.get_random_configs(n, False)
            goals_q, goals_z, _ = self.get_random_configs(n, False)
        elif mode == "CCi":
            starts_q, starts_z, _ = self.get_random_configs(n, False)
            goals_q, goals_z, _ = self.get_random_configs(n, False)
            starts_q, starts_r = self.get_interp_config(alpha, starts_q, starts_z, goals_q, goals_z, False)

        elif mode == "RR":
            assert alpha == 1.0, "CC does not support curriculum"
            starts_q, starts_z, starts_r = self.get_random_configs(n, True)
            goals_q, goals_z, _ = self.get_random_configs(n, True)
        elif mode == "RRi":
            starts_q, starts_z, _ = self.get_random_configs(n, True)
            goals_q, goals_z, _ = self.get_random_configs(n, True)
            starts_q, starts_r = self.get_interp_config(alpha, starts_q, starts_z, goals_q, goals_z, True)

        elif mode == "RC":
            assert alpha == 1.0, "CC does not support curriculum"
            starts_q, starts_z, starts_r = self.get_random_configs(n, True)
            goals_q, goals_z, _ = self.get_random_configs(n, False)
        elif mode == "RCi":
            starts_q, starts_z, _ = self.get_random_configs(n, True)
            goals_q, goals_z, _ = self.get_random_configs(n, False)
            starts_q, starts_r = self.get_interp_config(alpha, starts_q, starts_z, goals_q, goals_z, True)

        elif mode == "rvC":
            goals_q, goals_z, _ = self.get_random_configs(n, False)
            starts_q, goals_feat = self.get_reverse_curriculum_configs(alpha, goals_q)
            starts_r = starts_q
        elif mode == "rvR":
            goals_q, goals_z, _ = self.get_random_configs(n, True)
            starts_q, goals_feat = self.get_reverse_curriculum_configs(alpha, goals_q)
            starts_r = starts_q

        else:
            raise Exception(f"mode {mode} not defined")

        if goals_feat is None:
            goals_feat = []
            for i in range(goals_q.shape[0]):
                goals_feat.append(self.goal_feat_map(goals_q[i]))
            goals_feat = np.stack(goals_feat)

        return starts_q, starts_r, goals_q, goals_feat

    def get_random_configs(self, n, rndDrop=True):
        if not hasattr(self, "configs"):
            self.load_configs()
        if rndDrop and not hasattr(self, "rndDropConfigs"):
            self.load_rndDropConfigs()

        if rndDrop:
            configs = self.rndDropConfigs
        else:
            configs = self.configs

        self.ensure_config_embeddings(configs)
        self.ensure_config_refs(configs)

        i = np.random.randint(configs.q.shape[0], size=n)
        return configs.q[i], configs.z[i], configs.r[i]

    def get_interp_config(self, alpha, starts_q, starts_z, goals_q, goals_z, rndDrop):
        if rndDrop:
            configs = self.rndDropConfigs
        else:
            configs = self.configs

        if not hasattr(configs, "tree"):
            self.create_config_KDTree(configs)
        tree: KDTree = configs.tree

        assert alpha <= 1.0
        assert starts_q.ndim == 2

        interp_q = np.empty(starts_q.shape)
        interp_r = np.empty(starts_q.shape)

        for i in range(starts_q.shape[0]):
            t_interp = np.random.uniform(0.0, alpha)

            z = t_interp * starts_z[i] + (1.0 - t_interp) * goals_z[i]
            d, nearest = tree.query(z, k=2)

            interp_q[i] = configs.q[nearest[0]]
            interp_r[i] = configs.r[nearest[0]]
            d_q = np.linalg.norm(interp_q[i] - goals_q[i])
            if d_q < 1e-10:
                interp_q[i] = configs.q[nearest[1]]
                interp_r[i] = configs.r[nearest[1]]

            if (i % 100) == 0:
                print("\r   interpolation processing: ", i, "/", starts_q.shape[0], end="")

        print(f"\n-- interpolated starts: {interp_q.shape[0]} (alpha: {alpha})")

        return interp_q, interp_r

    def set_reverse_curriculum_helper_gym(self, gym):
        self.helper_gym = gym

    def get_reverse_curriculum_configs(self, alpha, goals_q):
        goals_feat = []

        # assert self.cfg.T_curr>0, 'reverse currriculum does not make sense without T_curr'
        steps = int(alpha * self.cfg.nominal_time_limit / 0.05)
        if steps == 0:
            steps += 1

        starts_q = np.empty(goals_q.shape)
        for i in range(goals_q.shape[0]):
            goals_feat.append(self.goal_feat_map(goals_q[i]))
            starts_q[i] = self.get_reverse_curriculum_config(self.helper_gym, goals_q[i], goals_feat[i], steps=steps)

            if (i % 100) == 0:
                print("\r   reverse curriculum processing: ", i, "/", goals_q.shape[0], end="")

        print(f"\n-- created reverse curriculum starts: {len(goals_feat)} (alpha: {alpha}, steps: {steps})")

        return starts_q, np.stack(goals_feat)

    def get_reverse_curriculum_config(self, helper_env: MujocoGym, goal_q, goal_feat, steps, view=False):
        assert helper_env.num_scenes == 1
        helper_env.set_starts_goals(np.atleast_2d(goal_q), np.atleast_2d(goal_q), np.atleast_2d(goal_q), np.atleast_2d(goal_feat))

        # tim: `use_warp` no longer exists
        if helper_env.sim.warp_worlds > 0:
            helper_env.sim.data.time.zero_()
        else:
            helper_env.sim.data.time = 0.0
        helper_env.sim.ctrl_time = 0.0
        obs, info = helper_env.reset()

        if view:
            helper_env.sim.view_speed = 0.1
        for t in range(steps):
            if view:
                helper_env.sim.C.view(True, f"rc_sim step {t}")
            action = 1.0 * torch.randn(helper_env.num_scenes, helper_env.action_dim, device=helper_env.device)
            next_obs, reward, terminated, truncated, info = helper_env.step(action)
            obs = next_obs
        if view:
            helper_env.sim.C.view(True, f"rc_sim step {steps}")

        sample_q = helper_env.qpos.detach().cpu().numpy()
        return sample_q

    def get_random_task(self, goal_not_on_table=True):
        if not hasattr(self, "configs"):
            self.load_configs()

        table_id = -1
        if goal_not_on_table and "table" in self.supportObjs:
            table_id = self.supportObjs["table"]

        while True:
            task = np.random.randint(self.configs.q.shape[0] - 1, size=2)
            if task[0] >= task[1]:
                task[0] += 1
            if goal_not_on_table and table_id in self.configs.contacts[task[1]]:
                continue
            break
        return task.tolist()

    def get_interp_task(self, alpha, goal_not_on_table=True):
        if not hasattr(self, "configs"):
            self.load_configs()

        if not hasattr(self.configs, "tree"):
            self.create_config_KDTree(self.configs)

        tree: KDTree = self.configs.tree

        task = self.get_random_task(goal_not_on_table)

        assert alpha <= 1.0

        t_interp = np.random.uniform(0.0, alpha)

        z0 = self.configs.z[task[0]]
        z1 = self.configs.z[task[1]]
        z = t_interp * z0 + (1.0 - t_interp) * z1
        d, i = tree.query(z, k=2)
        # print(f'eps task: org: {task} org-dist: {np.linalg.norm(z0-z1)} alpha: {alpha}, near: {i} distance: {d}')
        # j = np.random.choice(i)
        # while(j == task[1]):
        #     j = np.random.choice(i)
        j = i[0]
        if j == task[1]:
            j = i[1]
        task[0] = int(j)
        return task

    def getStartsGoals_forTesting(self, n=1, verbose=0):
        if not hasattr(self, "configs"):
            self.load_configs()

        self.ensure_config_refs(self.configs, verbose=verbose)

        starts_q, starts_r, goals_q, goals_feat = [], [], [], []

        id2frame = self.C.getFrames()

        for j in range(n):
            task = [j, j + 1]

            starts_q.append(self.configs.q[task[0]])
            starts_r.append(self.configs.r[task[0]])
            goals_q.append(self.configs.q[task[1]])
            goals_feat.append(self.goal_feat_map(self.configs.q[task[1]]))

            if (j % 100) == 0:
                print("\r   processing: ", j, "/", n, end="")

        print(f"\n-- created testing starts+goals: {len(starts_q)}")

        return np.stack(starts_q), np.stack(starts_r), np.stack(goals_q), np.stack(goals_feat)

    def getStartsGoals_fromRandomTasks(self, n=10_000, alpha=None):
        if not hasattr(self, "configs"):
            self.load_configs()
        self.ensure_config_refs(self.configs, verbose=False)

        starts_q, starts_r, goals_q, goals_feat = [], [], [], []

        for i in range(n):
            if alpha is None:
                task = self.get_random_task()
            else:
                task = self.get_interp_task(alpha)
            starts_q.append(self.configs.q[task[0]])
            starts_r.append(self.configs.r[task[0]])
            goals_q.append(self.configs.q[task[1]])
            goals_feat.append(self.goal_feat_map(self.configs.q[task[1]]))

            if (i % 100) == 0:
                print("\r   processing: ", i, "/", n, end="")

        print(f"\n-- created rand starts+goals: {len(starts_q)} (alpha: {alpha})")

        return np.stack(starts_q), np.stack(starts_r), np.stack(goals_q), np.stack(goals_feat)

    def getStartsGoals_fromTrajData(self, alpha):
        if not hasattr(self, "traj_data"):
            self.load_TrajectoryData(build_observations=False)

        S = self.traj_data.state.shape
        T = S[1]
        if alpha <= 1.0:
            T0 = int((1.0 - alpha) * T)
        else:
            T0 = 0
        starts_q, starts_v, goals_q, goals_v, goals_feat = [], [], [], [], []

        for i in range(S[0]):
            for t in range(T0, S[1]):
                x = self.sim.to_state(self.traj_data.goal[i, t])
                goals_q.append(x.qpos.copy())
                goals_v.append(x.qvel.copy())
                goals_feat.append(self.goal_feat_map(x.qpos))

                x = self.sim.to_state(self.traj_data.state[i, t])
                starts_q.append(x.qpos.copy())
                starts_v.append(x.qvel.copy())

        # goals = self.traj_data.goal[:,T0:,:]
        # starts = self.traj_data.state[:,T0:,:]

        # # WATCH - static (i.e. zero velocity) resets:
        # S = starts.shape
        # for i in range(S[0]):
        #     if (i%100)==0:
        #         print('\r   processing: ', i, '/', S[0], end='')
        #     for t in range(S[1]):
        #         s = self.sim.to_state(starts[i,t])
        #         s.time *= 0.
        #         # s.qvel *= 0.
        #         s.act *= 0.
        #         starts[i,t] = s.as_vector()

        # starts = starts.reshape(-1,S[2])
        # goals = goals.reshape(-1,S[2])

        print(f"\n-- created traj starts+goals: {len(starts_q)} (alpha: {alpha})")

        return np.stack(starts_q), np.stack(goals_q), None  # starts, goals

    def load_configs(self, verbose=1):
        self.configs = ConfigsFile(self.cfg.files.configs)
        self.supportObjs = {}
        for f_name in self.configs.supports:
            id = self.C.getFrame(f_name).ID
            self.supportObjs[f_name] = id
        if self.configs.q.shape[1] + 4 == self.C.getJointDimension():  # quaternions missing
            self.configs.q = np.hstack((self.configs.q, np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.configs.q.shape[0], 1))))
        if self.configs.q.shape[1] - 3 == self.C.getJointDimension():  # COM dofs too much
            self.configs.q = self.configs.q[:, :-3]
        assert self.configs.q.shape[1] == self.C.getJointDimension()

        # in case the frame ID changed (the config file was different/sorted differently)
        if "frame_names" in self.configs.manifest:
            file_names = self.configs.manifest["frame_names"]
            # C_names = self.C.getFrameNames()
            for j, c in enumerate(self.configs.contacts):
                # print(c, [file_names[i] for i in c])
                c = [self.C.getFrame(file_names[i]).ID for i in c]
                self.configs.contacts[j] = np.array(c, dtype="int32")
                # print(c, [C_names[i] for i in c])

        if verbose > 0:
            print("-- loaded configs:", self.configs.q.shape[0])

    def load_rndDropConfigs(self, verbose=1):
        self.rndDropConfigs = ConfigsFile(self.cfg.files.rndDropConfigs)
        assert self.rndDropConfigs.q.shape[1] == self.C.getJointDimension()
        if verbose > 0:
            print("-- loaded configs:", self.rndDropConfigs.q.shape[0])

    def ensure_config_embeddings(self, configs):
        if not hasattr(configs, "z"):
            n = configs.q.shape[0]
            z = self.interp_embedding(configs.q[0], None, self.configs.supports)
            configs.z = np.empty((n, z.size))
            for i in range(n):
                configs.z[i] = self.interp_embedding(configs.q[i], None, self.configs.supports)
                if (i % 1000) == 0:
                    print("\r   computing embeddings: ", i, "/", n, end="")
            print()

    def ensure_config_refs(self, configs, verbose=0):
        if not hasattr(configs, "r"):
            if self.cfg.force_reference_scale < 0.0:
                frames = self.C.getFrames()
                joints = self.C.getJoints()
                configs.r = configs.q.copy()
                (n, nq) = configs.q.shape
                for i in range(n):
                    r_i = configs.r[i]
                    self.C.setJointState(r_i)
                    if verbose > 0:
                        self.C.view(True, f"before {self.configs.forces[i]}")
                    # get gravity forces
                    F0 = np.zeros((nq,))
                    for f in frames:
                        m, com, I = f.getInertia()
                        if m > 0.0:
                            if np.linalg.norm(com) > 1e-10:
                                J = f.get_Jacobian_pos(f.trans_rel2world(com))
                            else:
                                J = f.get_Jacobian_pos()
                            F0 += J[2, :] * 9.81 * m
                    # get contact forces
                    Fc = np.zeros((nq,))
                    c = self.configs.contacts[i]
                    nc = c.size // 2
                    c = c.reshape((nc, 2))
                    f = self.configs.forces[i].reshape(nc, 3, 3)  # poa force torque
                    for j in range(nc):
                        a, b = c[j]
                        poa, force = f[j, 0, :], f[j, 1, :]
                        if verbose > 0:
                            print(f" contact {frames[a].name}--{frames[b].name} with force: {force}, poa: {poa}")
                        J = frames[a].get_Jacobian_pos(poa)
                        Fc += J.T @ force
                        J = frames[b].get_Jacobian_pos(poa)
                        Fc -= J.T @ force

                    first_joint_type = joints[0].getJointType()
                    last_joint_type = joints[-1].getJointType()
                    if first_joint_type == ry.JT.free:  # this is a free floating robot
                        F = F0 - Fc
                        F[:7] = 0.0
                        if verbose > 0:
                            print("first joint is free floating robot - zeroing forces")

                    if last_joint_type == ry.JT.free:  # this is a free floating object
                        F = F0 - Fc
                        F[-7:] = 0.0
                        if verbose > 0:
                            print("last joint is free floating object - zeroing forces")

                    # get kp's
                    Kp = np.ones((nq,))
                    for j in joints:
                        j_idx = j.getJointQIndex()
                        atts = j.getAttributes()
                        kp = None
                        if "kp" in atts:
                            kp = atts["kp"]
                            Kp[j_idx] = kp
                        if verbose > 2:
                            print(f"joint {j.name} [{j_idx}] has kp: {kp}")

                    r_i += 1.0 * (F / Kp)
                    if verbose > 0:
                        self.C.setJointState(r_i)
                        self.C.view(True, f"after {c} {self.configs.forces[i]}")
                    configs.r[i] = r_i  # is redundant! r_i is a reference

                    if (i % 1000) == 0:
                        print("\r   computing force-based refs: ", i, "/", n, end="")

            elif self.cfg.force_reference_scale > 0.0:
                id2frame = self.C.getFrames()
                configs.r = configs.q.copy()
                n = configs.q.shape[0]
                for i in range(n):
                    r_i = configs.r[i]
                    c = self.configs.contacts[i]
                    nc = c.size // 2
                    f = self.configs.forces[i].reshape(nc, 3, 3)  # poa force torque
                    self.C.setJointState(r_i)
                    if verbose > 0:
                        print("##", [id2frame[j].name for j in c[1::2]], f[:, 1, :])
                        self.C.view(True, f"start config {i}")
                    for j in range(nc):
                        J = id2frame[c[1 + 2 * j]].get_Jacobian_pos(f[j, 0, :])
                        r_i += self.cfg.force_reference_scale * (J.T @ f[j, 1, :])
                        if verbose > 0:
                            self.C.setJointState(r_i)
                            self.C.view(True, f"start config {i} force {j}")
                    configs.r[i] = r_i

                    if (i % 1000) == 0:
                        print("\r   computing force-based refs: ", i, "/", n, end="")
            else:
                configs.r = configs.q
            print()

    def create_config_KDTree(self, configs):
        self.ensure_config_embeddings(configs)

        mem0 = proc_mem()
        configs.tree = KDTree(data=configs.z)
        print(f"   KD tree built, mem: {proc_mem() - mem0:.2f}GB")

        return configs.tree

    def load_episodes(self, verbose=1):
        self.episodes = EpisodesFile(self.cfg.files.episodes)
        self.episodes.tau_step = self.episodes.manifest["tau_step"]
        if self.episodes.start.shape[1] + 4 == self.q_dim:  # quaternions missing
            self.episodes.start = np.hstack((self.episodes.start, np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.episodes.start.shape[0], 1))))
            self.episodes.goal = np.hstack((self.episodes.goal, np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.episodes.goal.shape[0], 1))))
        assert self.episodes.manifest["num_episodes"] == self.episodes.start.shape[0]
        assert self.episodes.tau_step == self.cfg.Gym.tau_step
        assert self.episodes.start.shape[1] == self.q_dim
        assert self.episodes.goal.shape[1] == self.q_dim
        if verbose > 0:
            print("-- loaded episodes:", self.episodes.start.shape[0])

    def load_TrajectoryData(self, build_observations=True, verbose=1):
        self.load_episodes()
        S = self.episodes.state.shape

        self.traj_data = TrajectoryData()
        self.traj_data.state = self.episodes.state
        self.traj_data.action = self.episodes.action
        self.traj_data.goal = np.empty((S[0], S[1], S[2]))
        self.traj_data.time_to_go = np.empty((S[0], S[1], 1))

        for i in range(S[0]):
            if (i % 100) == 0:
                print("\r   processing: ", i, "/", S[0], end="")
            Ti = S[1]
            goal = self.traj_data.state[i, -1]
            self.traj_data.goal[i, :] = goal
            for t in range(Ti):
                self.traj_data.time_to_go[i, t, 0] = (Ti - 1 - t) * self.episodes.tau_step

        if build_observations:
            # save the current gym goal
            if hasattr(self.gym, "goal_feat"):
                org_goal_feat = self.gym.goal_feat.copy()
            else:
                org_goal_feat = None
                # just to have some goal set:
                x = self.gym.sim.to_state(self.traj_data.state[0, 0])
                _, self.gym.goal_feat = self.gym.observation_fct(x, without_goal=True)

            # resize the observation array
            x = self.gym.sim.to_state(self.traj_data.state[0, 0])
            obs, _ = self.gym.observation_fct(x)
            self.traj_data.observation = np.empty((S[0], S[1], obs.size))

            for i in range(S[0]):
                if (i % 100) == 0:
                    print("\r   processing: ", i, "/", S[0], end="")
                Ti = S[1]
                # set gym goal to traj goal
                goal = self.traj_data.state[i, -1]
                x = self.gym.sim.to_state(goal)
                _, self.gym.goal_feat = self.gym.observation_fct(x, without_goal=True)
                for t in range(Ti):
                    # get the state observation (which may be relative to the goal)
                    state = self.traj_data.state[i, t]
                    x = self.gym.sim.to_state(state)
                    obs, _ = self.gym.observation_fct(x)
                    self.traj_data.observation[i, t] = obs

            # restore the current gym goal
            self.gym.goal_feat = org_goal_feat

        print(f"\n-- loaded trajectories: {self.traj_data.state.shape[0]}")

    def evaluate_policy(self, env: MujocoGym, policy, video_C=None, num_episodes=20, view=False, make_video=False, collect_histograms=False, verbose=2, add_perturbations=False):
        assert not (view and make_video), "cannot view and make video at the same time currently"

        print("== evaluating policy...")

        if view or make_video:
            env.sim.view_speed = 1.0
            F_goals, F_obj = [], []
            for f in video_C.getFrames():
                if f.name.endswith("obj_goal"):
                    F_goals.append(f)
                if f.name.endswith("obj"):
                    F_obj.append(f)

        if make_video:
            video_C.view(False)
            env.sim.view_speed = 1.0
            env.sim.saved_images = []
            rgb = video_C.get_viewer().getRgb()
            time.sleep(0.1)
            rgb = video_C.get_viewer().getRgb()
            V = video_gen.VideoGenerator(input_framerate=50)
            V.add([rgb] * 10)

        if collect_histograms:
            actions = []
            obs = []

        env.total_eps, env.total_steps, env.total_reward, env.total_succ, env.total_terminated = 0.0, 0.0, 0.0, 0.0, 0.0

        env.scene_needs_reset[:] = True
        n_ep_started = env.num_scenes
        n_ep = 0
        n_succ = 0
        tot_reward, tot_steps = 0.0, 0.0
        active = torch.full((env.num_scenes,), True, dtype=torch.bool, device=env.device)
        ep_reward = torch.zeros((env.num_scenes,), dtype=torch.float32, device=env.device)
        ep_steps = torch.zeros((env.num_scenes,), dtype=torch.float32, device=env.device)

        while torch.any(active):  # n_ep<num_episodes:
            state, info = env.auto_reset()
            if env.num_scenes == 1:
                env.sim.view_text = f"episode {n_ep}, acc. reward {ep_reward.item():.3f},"

            if view or make_video:
                # reposition goal frames for illustration
                goal_glob = env.goal_q.detach().cpu().numpy().copy()
                for th in range(len(F_goals)):
                    F_goals[th].setRelativePosition(goal_glob[th, -7:-4])
                    F_goals[th].setRelativeQuaternion(goal_glob[th, -4:])
                for f in F_obj:
                    f.setColor([1.0, 0.5, 0.0])

            # add perturbation?
            if add_perturbations and (n_ep // 10) % 2:
                env.sim.view_text += " WITH PERTURBATIONS (RED),"
                if np.random.random() < 0.2:
                    delta = np.random.normal(env.num_scenes, 3)
                    env.qvel[:, -6:-3] += 0.5 / np.linalg.norm(delta) * delta
                    for f in F_obj:
                        f.setColor([1.0, 0.0, 0.0])

            # get action - account for HER replay buffer observations..
            if hasattr(policy, "replay_buffer") and hasattr(policy.replay_buffer, "her_ratio"):
                obs_dict = {
                    "observation": _to_np(env.observation),
                    "achieved_goal": _to_np(env.feat),
                    "desired_goal": _to_np(env.goal_feat),
                }
                action, _ = policy.predict(obs_dict, deterministic=True)
            elif hasattr(policy, "policy"):  # this is a sb3 agent
                action, _ = policy.predict(state.numpy(force=True), deterministic=True)
                action = torch.tensor(action, dtype=torch.float32, device=env.device)
            else:
                action, _ = policy.predict(state, deterministic=True)

            if verbose > 2:
                print(ep_steps, f"{env.sim.ctrl_time:.3f}", state, action)

            # step
            state, reward, terminated, truncated, info = env.step(action)
            ep_finished = torch.logical_or(terminated, truncated)
            ep_reward += reward
            ep_steps += 1.0

            for k in range(env.num_scenes):
                if ep_finished[k] and active[k]:  # only when active!
                    if verbose > 1:
                        print(f"episode {n_ep}, terminated: {terminated[k]}, acc. reward: {ep_reward[k]} steps: {ep_steps[k]}")

                    if terminated[k]:
                        if ep_reward[k] < 0.999:
                            print("-- WARNING: success should be equivalent to termimated and equivalent to eq_reward>1.")
                        if view:
                            if env.num_scenes == 1:
                                video_C.view(False, f"episode {n_ep}, acc. reward {ep_reward.item():.3f}, simulation t: {env.sim.ctrl_time:6.3f} -- SUCCESS")
                            time.sleep(0.2)
                        n_succ += 1

                    if make_video:
                        if terminated[k] and ep_steps[k] >= 5:
                            video_C.view(False, f"episode {n_ep}, acc. reward {ep_reward.item():.3f}, simulation t: {env.sim.ctrl_time:6.3f} -- SUCCESS", offscreen=True)
                            rgb = env.sim.C.get_viewer().getRgb()
                            env.sim.saved_images.append([rgb] * 10)
                        V.add(env.sim.saved_images)
                        env.sim.saved_images = []

                    tot_reward += ep_reward[k]
                    tot_steps += ep_steps[k]
                    ep_reward[k] = 0.0
                    ep_steps[k] = 0.0
                    n_ep += 1

                    if n_ep_started >= num_episodes:  # we don't want to reuse that thread/scene to collect more data
                        active[k] = False
                    else:
                        n_ep_started += 1  # next loop will restart this 'thread/scene'

            if collect_histograms:
                actions.append(action)
                obs.append(state)

        print(
            f"== env stats: #eps: {env.total_eps}, succ rate: {env.total_succ / env.total_eps}, avg reward: {env.total_reward / env.total_eps}, avg terminated: {env.total_terminated / env.total_eps}, avg steps: {env.total_steps / env.total_eps}"
        )
        print(f"== local stats: #eps: {n_ep}, succ rate: {n_succ / n_ep}, avg reward: {tot_reward / n_ep}, avg steps: {tot_steps / n_ep}")

        if collect_histograms:
            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            # tim: `num_scenes` is undefined here
            n = len(actions) * num_scenes
            # tim: this should fail since its trying to numpy stack torch tensors
            actions = np.stack(actions).reshape(n, -1)
            ax0.hist(actions, 21, histtype="step", stacked=True, fill=False)
            ax0.set_title("actions")
            ax0.legend(labels=[f"{i}" for i in range(actions.shape[1])], prop={"size": 7})

            obs = np.stack(obs).reshape(n, -1)
            ax1.hist(obs, 21, histtype="step", stacked=True, fill=False)
            ax1.set_title("observations")
            ax1.legend(labels=[f"{i}" for i in range(obs.shape[1])], prop={"size": 7})
            for i in range(obs.shape[1]):
                A, _ = np.histogram(obs[:, i], range=(-1.0, 1.0), bins=20)
                s = np.std(obs[:, i])
                print(f"index {i:2}: {s} {A}")

            fig.tight_layout()
            plt.show()

        # return env.total_succ/env.total_eps, env.total_reward/env.total_eps, env.total_terminated/env.total_eps, env.total_steps/env.total_eps
        return n_succ / n_ep, tot_reward / n_ep, n_succ / n_ep, tot_steps / n_ep

    def get_TrajCloneBuffer(self, phase) -> TrajCloneBuffer:
        if not hasattr(self.W, "traj_data"):
            self.W.load_TrajectoryData()

        S = self.W.traj_data.state.shape
        T = S[1]
        if phase <= 1.0:
            T0 = int((1.0 - phase) * T)
        else:
            T0 = 0

        clone_buffer = TrajCloneBuffer()
        time_to_go = self.W.traj_data.time_to_go[:, T0:, :]
        o_dim = self.W.traj_data.observation.shape[2]

        clone_buffer.observation = self.W.traj_data.observation[:, T0:, :]
        clone_buffer.action = self.W.traj_data.action[:, T0:, :]
        clone_buffer.Return = np.empty((S[0], T - T0, 1))
        clone_buffer.Gamma = np.zeros((S[0], T - T0, 1))  # ZERO!
        clone_buffer.next_observation = np.empty((S[0], T - T0, o_dim))

        for i in range(S[0]):
            Ti = clone_buffer.observation.shape[1]

            # set next_state_obs to final state
            final_obs = clone_buffer.observation[i, -1]
            clone_buffer.next_observation[i, :] = final_obs

            for t in range(Ti):
                clone_buffer.Return[i, t, 0] = 1.0 - self.W.gym.cfg.cost_const * time_to_go[i, t, 0]

        for key, x in clone_buffer.__dict__.items():
            clone_buffer.__dict__[key] = x.reshape(-1, x.shape[-1])

        print("-- created clone_buffer:")
        for key, x in clone_buffer.__dict__.items():
            print("  ", key, x.shape)

        return clone_buffer
