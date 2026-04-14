import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from .objects.rattle_cube import CubeObject, CUBE_MATERIALS

VALID_OBJECT_TYPES = tuple(CUBE_MATERIALS.keys())

CONTACT_CUBE_TABLE = "cube_table"

_REWARD_SCALE = 99.5

REWARD_MODES = ("sustained", "impact")


class BangingEnv(ManipulationEnv):

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        object_type="plastic_cube",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        use_contact_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        reward_scale=_REWARD_SCALE,
        reward_mode="sustained",
    ):
        assert object_type in VALID_OBJECT_TYPES, \
            f"object_type must be one of {VALID_OBJECT_TYPES}, got {object_type!r}"
        assert reward_mode in REWARD_MODES, \
            f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}"

        self.object_type = object_type

        self.table_full_size = table_full_size
        self.table_friction  = table_friction
        self.table_offset    = np.array((0, 0, 0.8))

        self.use_object_obs  = use_object_obs
        self.use_contact_obs = use_contact_obs

        self.reward_scale = reward_scale
        self.reward_mode  = reward_mode

        self.cube_body_id       = None
        self.cube_geom_id       = None
        self.table_geom_id      = None

        self._outer_events    = {}
        self._total_outer_fn  = 0.0
        self._impact_outer_fn = 0.0

        self._current_contacts = {
            CONTACT_CUBE_TABLE: [],
        }

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _post_action(self, action):
        self._detect_contacts()
        return super()._post_action(action)

    def _detect_contacts(self):
        contact_force = np.zeros(6, dtype=np.float64)
        current_outer = {}

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            g1, g2  = contact.geom1, contact.geom2
            pair    = (min(g1, g2), max(g1, g2))
            mujoco.mj_contactForce(
                self.sim.model._model, self.sim.data._data, i, contact_force
            )
            fn = float(abs(contact_force[0]))

            if g1 == self.table_geom_id or g2 == self.table_geom_id:
                other = g2 if g1 == self.table_geom_id else g1
                if other == self.cube_geom_id:
                    current_outer[pair] = max(current_outer.get(pair, 0.0), fn)

        for pair in set(self._outer_events) - set(current_outer):
            del self._outer_events[pair]

        self._total_outer_fn  = 0.0
        self._impact_outer_fn = 0.0
        for pair, fn in current_outer.items():
            if pair not in self._outer_events:
                self._outer_events[pair] = {"n_steps": 0}
            self._outer_events[pair]["n_steps"] += 1
            self._total_outer_fn += fn
            if self._outer_events[pair]["n_steps"] == 1:
                self._impact_outer_fn += fn

        self._current_contacts = {
            CONTACT_CUBE_TABLE: [],
        }
        for pair, fn in current_outer.items():
            self._current_contacts[CONTACT_CUBE_TABLE].append({
                "wall":         "cube",
                "pair":         pair,
                "normal_force": fn,
                "is_new":       self._outer_events[pair]["n_steps"] == 1,
            })

    def get_contacts(self):
        return self._current_contacts

    def reward(self, action=None):
        return float(np.clip(
            self._total_outer_fn / self.reward_scale, 0.0, 1.0
        ))

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.cube = CubeObject(
            name="banging_cube",
            material=self.object_type,
            joints="default",
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube],
        )

        gripper = self.robots[0].gripper.get("right")
        if gripper is None:
            gripper = list(self.robots[0].gripper.values())[0]
        gripper_body = gripper.root_body

        weld = ET.Element("weld")
        weld.set("body1", gripper_body)
        weld.set("body2", self.cube.root_body)
        weld.set("relpose", "0 0 0.12 1 0 0 0")
        weld.set("solref", "0.002 1")
        self.model.equality.append(weld)

    def _setup_references(self):
        super()._setup_references()

        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.cube_geom_id = self.sim.model.geom_name2id(
            f"{self.cube.naming_prefix}cube"
        )
        self.table_geom_id = self.sim.model.geom_name2id("table_collision")

        self._augment_gripper_inertia()

    def _augment_gripper_inertia(self):
        """
        Lump welded cube mass + inertia into the gripper body so OSC gravity
        compensation and mass-matrix terms account for the cube. Parallel-axis
        shift + eigendecomposition into a principal inertia frame.

        Offset of cube COM in gripper frame is [0, 0, 0.12] to match the weld
        `relpose` in `_load_model`.
        """
        gripper = self.robots[0].gripper.get("right")
        if gripper is None:
            gripper = list(self.robots[0].gripper.values())[0]
        gripper_body_id = self.sim.model.body_name2id(gripper.root_body)

        full_side = 2.0 * self.cube.size
        m_c = self.cube.cube_density * (full_side ** 3)
        r_c = np.array([0.0, 0.0, 0.12])   # cube COM in gripper-body frame

        I_c_self = (1.0 / 6.0) * m_c * (full_side ** 2)
        I_c = np.eye(3) * I_c_self

        m_g     = float(self.sim.model.body_mass[gripper_body_id])
        ipos_g  = np.array(self.sim.model.body_ipos[gripper_body_id])
        iquat_g = np.array(self.sim.model.body_iquat[gripper_body_id])
        Idiag_g = np.array(self.sim.model.body_inertia[gripper_body_id])

        R_g = np.zeros(9)
        mujoco.mju_quat2Mat(R_g, iquat_g)
        R_g = R_g.reshape(3, 3)
        I_g = R_g @ np.diag(Idiag_g) @ R_g.T

        m_total = m_g + m_c
        com_total = (m_g * ipos_g + m_c * r_c) / m_total

        def parallel_axis(I, m, d):
            return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        I_total = (parallel_axis(I_g, m_g, ipos_g - com_total)
                   + parallel_axis(I_c, m_c, r_c - com_total))

        eigvals, eigvecs = np.linalg.eigh(I_total)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1.0

        iquat_total = np.zeros(4)
        mujoco.mju_mat2Quat(iquat_total, eigvecs.flatten())

        self.sim.model.body_mass[gripper_body_id]    = m_total
        self.sim.model.body_ipos[gripper_body_id]    = com_total
        self.sim.model.body_iquat[gripper_body_id]   = iquat_total
        self.sim.model.body_inertia[gripper_body_id] = eigvals

        # Cube is a separate free-floating body with its own mass; its gravity
        # pulls on the gripper via the weld constraint as an external load that
        # OSC gravity-comp does not see. Disable cube gravity so the gripper's
        # augmented inertia is the single source of truth for the cube's mass.
        # body_gravcomp=1.0 -> MuJoCo automatically applies an upward force on
        # this body to exactly cancel gravity.
        self.sim.model.body_gravcomp[self.cube_body_id] = 1.0

        mujoco.mj_setConst(self.sim.model._model, self.sim.data._data)

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return np.array(self.sim.data.body_xquat[self.cube_body_id])

            for s in [cube_pos, cube_quat]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        if self.use_contact_obs:
            modality = "contact"

            @sensor(modality=modality)
            def contact_force(obs_cache):
                return np.array([
                    float(np.clip(self._total_outer_fn / self.reward_scale, 0.0, 1.0))
                ])

            @sensor(modality=modality)
            def cube_table_contact_binary(obs_cache):
                return np.array([1.0 if self._current_contacts[CONTACT_CUBE_TABLE] else 0.0])

            @sensor(modality=modality)
            def cube_table_contact_count(obs_cache):
                return np.array([float(len(self._current_contacts[CONTACT_CUBE_TABLE]))])

            for s in [
                contact_force,
                cube_table_contact_binary, cube_table_contact_count,
            ]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        self._outer_events   = {}
        self._total_outer_fn = 0.0

        super()._reset_internal()

        gripper = self.robots[0].gripper.get("right")
        if gripper is None:
            gripper = list(self.robots[0].gripper.values())[0]
        gripper_body_id = self.sim.model.body_name2id(gripper.root_body)
        gripper_pos  = self.sim.data.body_xpos[gripper_body_id]
        gripper_quat = self.sim.data.body_xquat[gripper_body_id]

        cube_pos  = gripper_pos + np.array([0, 0, -0.12])
        cube_quat = gripper_quat.copy()
        cube_qpos = np.concatenate([cube_pos, cube_quat])

        self.sim.data.set_joint_qpos(self.cube.joints[0], cube_qpos)
        self.sim.data.set_joint_qvel(self.cube.joints[0], np.zeros(6))

        self.sim.forward()

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
