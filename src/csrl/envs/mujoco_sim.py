import mujoco
import mujoco.viewer
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class MujocoSim:
    """
    This is wrapping the Mujoco simulator without yet defining a gym environment.

    It assumes that the simulation is stepped with a high frequency (~1kHz), while the
    policy provides new controls with a lower frequency. The high frequency stepping is
    reads out a control buffer, which are explicit and continuously/smoothly varying motor
    references for each micro step. This control buffer needs to be re-filled before each
    multi-step.
    The policy output typically defines a spline (see poly_ref.py, but could also be B-spline)
    which is sampled with the high frequency to fill that control buffer.

    Futher, this wrapper handles qoffsets: When the free objects in the original configuration
    have parent (i.e. non-zero origin) frames, this implies a qpos offset. That's the case
    esp. when the configuration is a multi-scene configuration.
    """

    mj_steps = 0
    ctrl_time = 0.0
    ctrl_costs = 0.0

    # for inspection & rendering only
    view_speed = -1.0
    view_callback = None
    saved_qpos = None

    def __init__(
        self,
        xml_description: str,
        qpos_init: np.array,
        qpos_offset: np.array,
        tau_sim: float = 1e-3,
        warp_worlds: int = 0,
        use_mj_viewer: bool = False,
        debug_collisions: bool = False,
        viewer_camera_rpy=None,
    ):
        """
        Basic simulation class that wraps a mujoco simulator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qpos_offset = torch.as_tensor(qpos_offset, dtype=torch.float32, device=self.device)
        self.tau_sim = tau_sim
        self.warp_worlds = warp_worlds
        self.use_mj_viewer = use_mj_viewer
        self.debug_collisions = debug_collisions

        assert self.device.type == "cuda" or self.warp_worlds == 0, "can't have warp without cuda"

        if xml_description.startswith("<"):
            self.model = mujoco.MjModel.from_xml_string(xml_description)
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_description)

        # IMPORTANT: any writes to `model.opt.*` must happen BEFORE `mjw.put_model`
        self.model.opt.timestep = self.tau_sim
        if self.warp_worlds > 0:
            self.wp_model = mjw.put_model(self.model)
            self.data = mjw.make_data(self.model, nworld=warp_worlds, njmax=200)
        else:
            self.data = mujoco.MjData(self.model)

        assert self.model.nq == qpos_init.size, "mj model qpos size does not match the given qpos_init"

        # mostly not used, but still useful to double check and debug ry->mj conversion
        if self.use_mj_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            if viewer_camera_rpy is not None:
                self.viewer.cam.azimuth = -90.0 + viewer_camera_rpy[1] * 180.0 / 3.1416
                self.viewer.cam.elevation = 90.0 - viewer_camera_rpy[0] * 180.0 / 3.1416
                self.viewer.cam.distance = 1.0
                self.viewer.cam.lookat = [0.0, 0.0, 0.5]
        else:
            self.viewer = None

        # determine which qpos indices are controlled (actuated) joints
        self.ctrl_indices = []
        self.ctrl_vindices = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator(i).trnid[0]
            qid = self.model.joint(joint_id).qposadr
            vid = self.model.joint(joint_id).dofadr
            # name = self.model.joint(joint_id).name
            # print(name, qid)
            self.ctrl_indices.append(int(qid[0]))
            self.ctrl_vindices.append(int(vid[0]))
        assert len(self.ctrl_indices) == self.model.nu
        self.ctrl_indices = torch.as_tensor(self.ctrl_indices, dtype=torch.int32, device=self.device)
        self.ctrl_vindices = torch.as_tensor(self.ctrl_vindices, dtype=torch.int32, device=self.device)

        # set offset to zero if not provided
        if self.qpos_offset.shape[0] == 0:
            self.qpos_offset = torch.zeros(self.data.qpos.shape)
        else:
            if self.warp_worlds > 0:
                self.qpos_offset = torch.tile(self.qpos_offset, (self.warp_worlds, 1))
            assert self.qpos_offset.shape == self.data.qpos.shape, "qpos_offset has wrong shape"

        print(f"-- initialized MjSim with () dimensions qpos: {self.model.nq} qvel: {self.model.nv} ctrl: {self.model.nu}")

        # Cache stable torch views of data.{qpos,qvel,actuator_force,xpos,ctrl}.
        if self.warp_worlds > 0:
            self._wp_device = wp.get_cuda_device()
            if torch.cuda.is_available():
                # Use torch's current CUDA stream as Warp's stream for everything
                self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
            else:
                self._wp_stream = None
            # self._qpos_offset_wp = wp.from_torch(self.qpos_offset)
            # self._qpos_offset_t = wp.to_torch(self._qpos_offset_wp)
            self._ctrl_buffer_wp = None  # to be created on first pass!
            self._qpos_raw_t = wp.to_torch(self.data.qpos)
            self._qpos_t = self._qpos_raw_t - self.qpos_offset
            self._qvel_t = wp.to_torch(self.data.qvel)
            self._act_t = wp.to_torch(self.data.actuator_force)
            self._qpos_array_id = id(self.data.qpos)
            self._qvel_array_id = id(self.data.qvel)
            self._act_array_id = id(self.data.actuator_force)
            self._step_graph = None
            self._step_graph_key = None
        else:
            self._wp_stream = None

        if self.warp_worlds > 0:
            qpos = np.tile(qpos_init, (self.warp_worlds, 1))
            self.set_state(torch.as_tensor(qpos, dtype=torch.float32, device=self.device))
        else:
            self.set_state(torch.as_tensor(qpos_init, dtype=torch.float32, device=self.device))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def rollout(self, ctrl_buffer: torch.tensor) -> None:
        np_ctrl_buffer = None
        if self.warp_worlds > 0:
            if self._ctrl_buffer_wp is None:  # first pass
                self._ctrl_buffer_wp = wp.from_torch(ctrl_buffer.contiguous())
            else:  # future passes: just copy, assuming has proper size
                self._copy_into_wp(self._ctrl_buffer_wp, ctrl_buffer)
        else:
            np_ctrl_buffer = ctrl_buffer.cpu().numpy()

        # CUDA-graph replay when there are no per-substep Python branches
        if self.warp_worlds > 0 and self.view_speed <= 0.0:
            self._rollout_graph(ctrl_buffer.shape[0], self._ctrl_buffer_wp)
        else:
            stream_ctx = wp.ScopedStream(self._wp_stream) if self._wp_stream is not None else _NullContext()
            with stream_ctx:
                view_steps = int(0.02 / self.tau_sim * self.view_speed)

                for k in range(ctrl_buffer.shape[0]):
                    if self.warp_worlds > 0:
                        wp.copy(self.data.ctrl, self._ctrl_buffer_wp[k])
                        mjw.step(self.wp_model, self.data)
                    else:
                        self.data.ctrl[:] = np_ctrl_buffer[k].reshape(-1)
                        mujoco.mj_step(self.model, self.data)

                    self.mj_steps += 1
                    self.ctrl_time += self.tau_sim
                    if self.warp_worlds > 0:
                        pass
                    else:
                        self.ctrl_costs += np.sum(np.square(self.data.actuator_force))

                    # -- the rest is visualization and logging

                    # storing the path
                    if self.saved_qpos is not None:
                        self.saved_qpos.append(self.data.qpos.copy())

                    if self.debug_collisions:
                        self._collision_debugging()

                    # visualization, and storing images
                    if self.view_speed > 0.0 and (self.mj_steps % view_steps == 0):
                        if self.use_mj_viewer:
                            self.viewer.sync()
                        if self.view_callback:
                            if self.warp_worlds > 0:
                                self.view_callback(self._qpos[0], self.ctrl_time, view_steps * self.tau_sim / self.view_speed)
                            else:
                                self.view_callback(self.data.qpos - self.qpos_offset.cpu().numpy(), self.ctrl_time, view_steps * self.tau_sim / self.view_speed)

                if self.warp_worlds > 0:
                    mjw.forward(self.wp_model, self.data)
                else:
                    mujoco.mj_forward(self.model, self.data)

        if self.warp_worlds == 0:  # needs copying from cpu mujoco to tensor on device
            self._qpos_t = torch.as_tensor(self.data.qpos, dtype=torch.float32, device=self.device) - self.qpos_offset
            self._qvel_t = torch.as_tensor(self.data.qvel, dtype=torch.float32, device=self.device)
            self._act_t = torch.as_tensor(self.data.actuator_force, dtype=torch.float32, device=self.device)
        else:
            self._qpos_t = self._qpos_raw_t - self.qpos_offset

    def _rollout_graph(self, num_steps, wp_ctrl_buffer):
        # tim: this cache key is invalidated every step since MujocoGym.step() now allocates a fresh ctrl_buffer via poly.sample_buffer() each call
        # this is most probably a source of quite a bit of slowdown
        cache_key = (num_steps, int(wp_ctrl_buffer.ptr), int(self.data.ctrl.ptr), int(self.data.qpos.ptr))
        if self._step_graph is None or self._step_graph_key != cache_key:
            # Warm-up
            wp.copy(self.data.ctrl, wp_ctrl_buffer[0])
            mjw.step(self.wp_model, self.data)  # mt: that step is double with the first below. Ok, because only 'recording' the compute?
            # tim: the warmup gets all the one-time work out of the capture that we don't do every step such as JIT compilations or allocations etc.
            # But it's true that we do two physics steps whenever we call this function for the first time or the cache key changes. If we get the caching right, this will not happen though besides the first step.
            mjw.forward(self.wp_model, self.data)
            wp.synchronize_device(self._wp_device)
            # Capture on Warp's native stream.
            with wp.ScopedCapture(device=self._wp_device, force_module_load=True) as cap:
                for k in range(num_steps):
                    wp.copy(self.data.ctrl, wp_ctrl_buffer[k])
                    mjw.step(self.wp_model, self.data)
                mjw.forward(self.wp_model, self.data)
            self._step_graph = cap.graph
            self._step_graph_key = cache_key

        # Make Warp's stream wait for torch
        if self._wp_stream is not None:
            warp_native = self._wp_device.stream
            warp_native.wait_stream(self._wp_stream)
        wp.capture_launch(self._step_graph)
        # Make torch's stream wait for Warp before subsequent torch ops read the results.
        if self._wp_stream is not None:
            self._wp_stream.wait_stream(self._wp_device.stream)

        # Update host accounting
        self.mj_steps += num_steps
        self.ctrl_time += num_steps * self.tau_sim

    def _collision_debugging(self):
        self.viewer.sync()
        # Check if any collisions are occurring in the current step
        if self.data.ncon > 0:
            print(f"\n--- {self.data.ncon} Collision(s) Detected ---")

            # Loop through all active contacts
            for i in range(self.data.ncon):
                contact = self.data.contact[i]

                # Get the IDs of the two colliding bodies
                body1_id = self.model.geom_bodyid[contact.geom1]
                body2_id = self.model.geom_bodyid[contact.geom2]

                # Resolve the Body IDs to their string names defined in the XML
                body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

                # Fallback if a body somehow doesn't have a name (e.g., the worldbody)
                body1_name = body1_name if body1_name else f"body_id_{body1_id}"
                body2_name = body2_name if body2_name else f"body_id_{body2_id}"

                contact_position = contact.pos
                contact_normal = contact.frame[:3]

                print(f"  Collision [{i}]: '{body1_name}' <---> '{body2_name}'")
                print(f"    Position (XYZ): {contact_position}")
                print(f"    Normal Vector : {contact_normal}")
                print(f"    Distance (Penetration): {contact.dist:.5f}")

    def _copy_into_wp(self, dst: wp.array, src) -> None:
        """Write src (np.ndarray or torch.Tensor) into dst (wp.array) without rebinding dst.
        For numpy: in-place H->D copy via .assign (no realloc).
        For torch: zero-copy view via wp.from_torch then D->D copy."""
        assert isinstance(src, torch.Tensor)
        assert src.is_contiguous()
        wp.copy(dst, wp.from_torch(src.detach()))

    def _to_np_cpu(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.is_contiguous()
        return x.detach().cpu().numpy()

    def set_state(self, qpos, qvel=None, act=None, kinematics_only=False):
        if self.warp_worlds == 0:
            qpos_cpu = self._to_np_cpu(qpos.reshape(-1) + self.qpos_offset)
            self.data.qpos[:] = qpos_cpu
            if self.view_speed > 0.0 and self.view_callback:
                self.view_callback(qpos_cpu, self.ctrl_time, 0.0)
            if qvel is None:
                self.data.qvel[:] = 0.0
            else:
                self.data.qvel[:] = self._to_np_cpu(qvel).reshape(-1)
            if act is None:
                self.data.actuator_force[:] = 0.0
            else:
                self.data.actuator_force[:] = self._to_np_cpu(act).reshape(-1)
            if kinematics_only:
                mujoco.mj_kinematics(self.model, self.data)
            else:
                mujoco.mj_forward(self.model, self.data)
        else:
            stream_ctx = wp.ScopedStream(self._wp_stream) if self._wp_stream is not None else _NullContext()
            with stream_ctx:
                qpos = qpos.reshape(self.warp_worlds, -1).to(self.device, non_blocking=True)
                self._copy_into_wp(self.data.qpos, qpos + self.qpos_offset)
                if self.view_speed > 0.0 and self.view_callback:
                    self.view_callback(qpos[0].detach().cpu().numpy(), self.ctrl_time, 0.0)

                if qvel is None:
                    self._qvel_t.zero_()
                else:
                    self._copy_into_wp(self.data.qvel, qvel.reshape(self.warp_worlds, -1))

                if act is None:
                    self._act_t.zero_()
                else:
                    self._copy_into_wp(self.data.actuator_force, act.reshape(self.warp_worlds, -1))

                if kinematics_only:
                    mjw.kinematics(self.wp_model, self.data)
                else:
                    mjw.forward(self.wp_model, self.data)

            # Debug: ensure no rebinding occurred (would invalidate cached torch views).
            if __debug__:
                assert id(self.data.qpos) == self._qpos_array_id, "data.qpos was rebound!"
                assert id(self.data.qvel) == self._qvel_array_id, "data.qvel was rebound!"
                assert id(self.data.actuator_force) == self._act_array_id, "data.actuator_force was rebound!"

        if self.warp_worlds == 0:  # needs copying from cpu mujoco to tensor on device - alternative, just set to qpos, qvel, act arguments!?
            self._qpos_t = torch.as_tensor(self.data.qpos, dtype=torch.float32, device=self.device) - self.qpos_offset
            self._qvel_t = torch.as_tensor(self.data.qvel, dtype=torch.float32, device=self.device)
            self._act_t = torch.as_tensor(self.data.actuator_force, dtype=torch.float32, device=self.device)
        else:
            self._qpos_t = self._qpos_raw_t - self.qpos_offset

        if self.use_mj_viewer:
            self.viewer.sync()
            # self.C.view(True)

    def get_Jacobian(self, body_ids):
        # body_id = mujoco.mj_name2id(self.model, 1, frame_name)
        # assert body_id>=0, f'frame name {frame_name} is not a mj body'

        Jpos = np.empty((len(body_ids), 3, self.model.nv))
        Jang = np.empty((len(body_ids), 3, self.model.nv))

        for i, body_id in enumerate(body_ids.reshape(-1)):
            pos = self.data.xpos[body_id]
            mujoco.mj_jac(self.model, self.data, Jpos[i], Jang[i], pos, body_id)

        if False:  # test
            self.C.setJointState(self._qpos)
            y, J = self.C.eval(ry.FS.position, [frame_name])
            print(np.linalg.norm(pos - y))
            print(Jpos, "\n", J)

        return Jpos, Jang

    def get_pts(self, pt_ids, world_id=0):
        if self.warp_worlds > 0:
            pts = wp.to_torch(self.data.xpos)[:, pt_ids]
            return pts.reshape((pts.shape[0] * pts.shape[1], pts.shape[2], 3))
        else:
            return self.data.xpos[pt_ids.cpu()]

    def get_pt_ids(self, pt_names):
        ids = []
        for name in pt_names:
            found = False
            for i in range(self.model.nbody):
                b_name = self.model.body(i).name
                if b_name.endswith(name):
                    ids.append(i)  # needs to be foud multiple time for multi-scene configs
                    found = True
            if not found:
                raise Exception(f"pt_name {name} not found in mujoco list of bodies")
        return ids
