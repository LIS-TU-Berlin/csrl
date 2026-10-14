import math
import time
from datetime import datetime

import numpy as np
import robotic as ry

from .helpers import process_cfg, set_env_seed, set_global_seeds
from .mujoco_gym import MujocoGym
from .mujoco_io import MujocoWriter
from .mujoco_sim import MujocoSim


class EnvCreator:
    """
    This is a problem specific class, which loads a configuration and provides
    helpers to create a (parallelized) sim and gym.
    It also provides stable-baselines compatible wrappers.
    """

    def __init__(self, cfg, tag_append=""):
        self.cfg = cfg
        self.cfg.tag = f"{datetime.now().strftime('%y%m%d-%H%M%S')}{tag_append}"

        print("-- setting up scenario", self.cfg.scenario)

        # -- take care of seeding
        self.seed = self.cfg.seed
        if self.seed == -1:
            self.seed = int(np.random.randint(1000, 9999))
            self.cfg.seed = self.seed  # ensures the seed is written in the config
        set_global_seeds(seed=self.seed, deterministic=self.cfg.deterministic)

        process_cfg(self.cfg)
        if "CSRL" in self.cfg:
            self.cfg.CSRL.files = self.cfg.files

        self.load_ry_config()
        self.Cgrid = self.C
        self.qpos_offset = np.zeros((0,))

    def get_gamma(self):
        return math.pow(0.5, self.cfg.Gym.tau_step / self.cfg.CSRL.discount_half_life)

    def load_ry_config(self):
        # -- load the robotic configuration
        self.C = ry.Config()
        self.C.addFile(self.cfg.files.scene_yaml)
        # remove non-simulation joints
        for f in self.C.getFrames():
            spec = f.asDict()
            if "simulation" in spec and spec["simulation"] == False and f.getJointType() != ry.JT.none:
                print("   deleting non-simulation joint:", f.name)
                f.setJoint(ry.JT.none)
        self.q_dim = self.C.getJointDimension()
        self.q_home = self.C.getJointState()

        # -- grab the observation pts
        self.obs_points = []
        if self.cfg.Gym.obs_points is not None:
            for name in self.cfg.Gym.obs_points:
                f = self.C.getFrame(name)
                assert f is not None, f"cannot find obs_points frame {name}"
                self.obs_points.append(f)

        # -- grab the goal feature pts, which define the goal feature map, and thereby the goal-conditioning in Gym
        self.goal_feat_pts_frames = []
        if self.cfg.Gym.goal_feat_points is not None:
            for name in self.cfg.Gym.goal_feat_points:
                f = self.C.getFrame(name)
                assert f is not None, f"cannot find goal_feat_pts frame {name}"
                self.goal_feat_pts_frames.append(f)

        print("-- loaded config q-dim:", self.C.getJointDimension(), "with joints", self.C.getJointNames())

    def create_grid_config(self, nx, ny, dist):
        self.Cgrid = ry.Config()
        for x in range(nx):
            for y in range(ny):
                base = self.Cgrid.addConfigurationCopy(self.C, f"{x * ny + y}_")  # the string adds a prefix to all frame names
                for i in range(base.ID, self.Cgrid.getFrameDimension()):
                    f = self.Cgrid.frame(i)
                    if f.getParent() is None:
                        p = f.getPosition()
                        p[0] += dist * (x - (nx - 1) / 2)
                        p[1] += dist * (y - (ny - 1) / 2)
                        f.setPosition(p)
        for f in self.Cgrid.getFrames():
            if "camera_init" in f.name:
                f.name = "camera_init"
                p = f.getPosition()
                p[0] += dist * (nx - 1) / 2
                p[1] += dist * (ny - 1) / 2
                f.setPosition(p)
                break

        return self.Cgrid

    def _get_free_objs(self, C):
        # determine free objects and their non-zero offset in qpos
        qpos_offset = np.zeros(C.getJointDimension())
        freeobjs = []
        for f in C.getFrames():
            parent = f.getParent()
            if parent == None or (f.getJointType() == ry.JT.free):
                if "mass" in f.asDict():
                    freeobjs.append(f)
                    if parent is not None:
                        assert parent.getParent() is None, "free joints need to have a root parent!"
                        offset = parent.getPosition()
                        quat = parent.getQuaternion()
                        assert np.linalg.norm(quat - np.array([1, 0, 0, 0])) < 1e-10
                        qid = f.getJointQIndex()
                        qpos_offset[qid : qid + 3] = offset
        return freeobjs, qpos_offset

    def get_xml(self, C, num_scenes=1):
        collide_exclude = self.cfg.Gym.collide_exclude
        if collide_exclude != [] and num_scenes > 1:
            tmp = []
            for i in range(num_scenes):
                tmp += [f"{i}_{a}" for a in collide_exclude]
            collide_exclude = tmp
        M = MujocoWriter(C, friction=self.cfg.Gym.friction, solref=self.cfg.Gym.solref, collide_exclude=collide_exclude)
        xml = M.str().decode("ascii")

        xml_file = self.cfg.files.scene_yaml.replace(".yml", "_conv.xml")
        with open(xml_file, "w") as fil:
            fil.write(xml)
        return xml

    def view_callback(self, C, q, ctrl_time, delta_wait):
        C.setJointState(q)
        C.view(False, f"simulation t: {ctrl_time:6.3f}")
        time.sleep(delta_wait)

    def create_Sim(self, Csim, num_scenes, warp_worlds):
        assert not (num_scenes > 1 and warp_worlds > 0), "for now: either warp_worlds OR num_scenes>1 (qpos_offset not treated)"

        if self.cfg.Gym.view_mj:
            self.C.view(False)
            cam_pose = self.C.get_viewer().getCamera_pose()
            q = ry.Quaternion()
            q.set(cam_pose[3:])
            cam_rpy = q.getRollPitchYaw()
        else:
            cam_rpy = None

        xml = self.cfg.files.scene_xml
        if xml == "auto":
            xml = self.get_xml(Csim, num_scenes)
        self.sim = MujocoSim(
            xml,
            Csim.getJointState(),
            self.qpos_offset,
            tau_sim=self.cfg.Gym.tau_sim,
            warp_worlds=warp_worlds,
            use_mj_viewer=self.cfg.Gym.view_mj,
            debug_collisions=self.cfg.Gym.debug_collisions,
            viewer_camera_rpy=cam_rpy,
        )
        self.sim.view_callback = lambda q, ctrl_time, delta_wait: self.view_callback(Csim, q, ctrl_time, delta_wait)

        return self.sim

    def create_Gym(self):
        return self.create_GridGym(self.cfg.Gym.num_scenes, 1)

    def create_GridGym(self, nx, ny, warp_worlds=-1, dist=5.0):
        if nx > 1 or ny > 1:
            self.Cgrid = self.create_grid_config(nx, ny, dist=dist)
        else:
            self.Cgrid = self.C
        C_org = self.C

        self.free_objs, self.qpos_offset = self._get_free_objs(self.Cgrid)

        self.C = self.Cgrid
        joint_limits = self.Cgrid.getJointLimits()

        if warp_worlds < 0:
            warp_worlds = self.cfg.Gym.warp_worlds
        _sim = self.create_Sim(self.Cgrid, num_scenes=nx * ny, warp_worlds=warp_worlds)

        self.gym = MujocoGym(_sim, cfg=self.cfg.Gym, num_scenes=nx * ny, terminal_limits=joint_limits)

        set_env_seed(self.gym, seed=self.seed)

        self.C = C_org
        return self.gym

    # This is awkward: we duplicate implementation of computing goal_features:
    # In the Gym environment this is done by evaluating the mj_sim ...
    # but here we do it based on a ry-Config. But they need to be consistent!
    def get_pt_feat(self, qpos, frames):
        assert qpos.ndim == 1

        q0 = self.C.getJointState()
        self.C.setJointState(qpos)
        feat = np.empty((len(frames), 3))
        for i, f in enumerate(frames):
            feat[i] = f.getPosition()
        # the last is the reference!
        for i in range(feat.shape[0]):
            feat[i] -= feat[-1]
        self.C.setJointState(q0)
        return feat[:-1].reshape(-1)

    def goal_feat_map(self, qpos):
        return self.get_pt_feat(qpos, self.goal_feat_pts_frames)

    # Also problematic: This is duplicating the definition of observations in gym
    def get_state_from_observation(self, obs):
        free_objs, qpos_offset = self._get_free_objs(self.C)

        n = self.C.getJointDimension()
        # print("free:", [f.name for f in free_objs], "offset: ", qpos_offset)
        # print("obs pts:", [f.name for f in self.obs_points])
        # print("goal pts:", [f.name for f in self.goal_feat_pts_frames])
        nf = len(free_objs)
        nobs_pts = 3 * (len(self.obs_points) - 1)
        ngoal_pts = 3 * (len(self.goal_feat_pts_frames) - 1)
        nctrl = n - 7 * nf
        nvel = n - nf

        # print("#obs:", obs.shape)
        # print("nctrl", nctrl)  # qpos
        # print("nvel", nvel)  # qvel
        # print("nerr", nctrl)  # qvel
        # print("npts", nobs_pts)
        # print("ngoal", ngoal_pts)

        q_ctrl = obs[:nctrl]
        qvel = obs[nctrl : nctrl + nvel]
        err = obs[nctrl + nvel : nctrl + nvel + nctrl]
        obs_pts = obs[nctrl + nvel + nctrl : nctrl + nvel + nctrl + nobs_pts]
        goal_pts = obs[nctrl + nvel + nctrl + nobs_pts :]

        q = np.concatenate((q_ctrl, obs_pts, np.array([1, 0, 0, 0])))  # HACK: obj orientation
        g = goal_pts + self.get_pt_feat(qpos=q, frames=self.goal_feat_pts_frames)

        return q, g

    def get_config_embedding(self, qpos, qvel, supports):  # scale_obj=2., scale_q=1., scale_vel=0., scale_d=0., d_sigma=.05): #c, scale_c=.0,
        if qpos.size > self.q_dim and (qpos.size % self.q_dim == 0):  # batch evaluation
            feat = []
            n = qpos.size // self.q_dim
            _qpos = qpos.reshape(n, -1)
            _qvel = qvel.reshape(n, -1)
            for i in range(n):
                feat.append(self.get_config_embedding(_qpos[i], _qvel[i], supports))
            return np.stack(feat)

        assert qpos.ndim == 1
        assert qpos.size == self.q_dim

        z = np.zeros((0,))

        # points or object position
        if "points" in self.cfg.metric:
            n = len(self.cfg.metric.points)
            self.C.setJointState(qpos)
            for f_name in self.cfg.metric.points:
                f = self.C.getFrame(f_name)
                pos = f.getPosition()
                z = np.concat((z, self.cfg.metric.scale_obj * pos))
        else:
            z = self.cfg.metric.scale_obj * qpos[-7:-4]

        # object orientation
        if self.cfg.metric.scale_ori > 0.0:
            quat = ry.Quaternion()
            quat.set(qpos[-4:])
            mat = quat.getMatrix()
            z = np.concat((z, self.cfg.metric.scale_ori * mat.reshape(-1)))

        # robot dofs
        if self.cfg.metric.scale_q > 0.0:
            z_q = qpos[:-3]
            z = np.concat((z, self.cfg.metric.scale_q * z_q))

        # object velocity
        if self.cfg.metric.scale_vel > 0.0 and qvel is not None:
            raise Exception("might be buggy")
            qvel = qvel[: self.q_dim]
            z_vel = qvel[-3:]
            # if scale_q>0.:
            #     z_vel = np.concat((z_vel, scale_q * qvel[:-3]))
            z = np.concat((z, cfg.scale_vel * z_vel))

        # concat mode binary indicator vector
        # z_c = np.any(c[:,None] == self.I, axis=0).astype(np.float64)

        if self.cfg.metric.scale_d > 0.0:
            # distance 'sigmoid'
            self.C.setJointState(qpos)
            D = np.array([self.C.eval(ry.FS.negDistance, [f_name, "obj"])[0][0] for f_name in supports])
            z_d = np.clip(1.0 + D / self.cfg.metric.d_sigma, 0.0, 1.0)  # negative distance clipped to [-1,0]
            # z_d = np.exp(D/d_sigma) # e^{-d/sig} = contact indicator in [0,1]
            z = np.concatenate((z, self.cfg.metric.scale_d * z_d))

        return z
