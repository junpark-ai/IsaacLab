# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, matrix_from_quat, quat_mul, quat_from_matrix

from spatialmath import UnitQuaternion, SE3
from roboticstoolbox import ctraj

@configclass
class EventCfg:
    # robot
    franka_joint_odd = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1357]"),
            # "position_range": (0, 0),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0, 0)
        }
    )
    franka_joint_even = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[246]"),
            # "position_range": (0, 0),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0, 0)
        }
    )

    # valve
    valve_joint = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve", joint_names=".*"),
            "position_range": (0, 0),
            # "position_range": (-0.5, 0.5),
            "velocity_range": (0, 0)
        }
    )
    valve_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve", body_names=".*"),
            "pose_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),  # 45 deg
                "yaw": (0, 0),  # 20 deg
                # "x": (-0.1, 0.1),
                # "y": (-0.1, 0.1),
                # "z": (-0.1, 0.1),
                # "roll": (-0.349, 0.349),
                # "pitch": (0, 1.57),  # 45 deg
                # "yaw": (-0.349, 0.349),  # 20 deg
            },
            "velocity_range": {}
        }
    )


@configclass
class FrankaValveGraspEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 7  # 6 pose + 1 gripper
    observation_space = 14  # Robot pose (6) + Valve pose (6) + Gripper (2)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.5, replicate_physics=True)

    # event
    events: EventCfg = EventCfg()

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
                # "panda_joint1": 0,
                # "panda_joint2": 0,
                # "panda_joint3": 0,
                # "panda_joint4": (-0.0698 - 3.0718) / 2,  # -1.5708
                # "panda_joint5": 0,
                # "panda_joint6": (3.7525 - 0.0175) / 2,  # 1.8675
                # "panda_joint7": 0,
                # "panda_finger_joint.*": 0.035,
            },
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=40.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=40.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # valve
    valve = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/kist-robot2/Desktop/round_valve_align/round_valve_align.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            # rot=(0.0, 0.0, 0.0, 1.0),
            rot=(1, 0, 0, 0),
            # rot=(0.6532815, 0.2705981, -0.2705981, -0.6532815),  # (w, x, y, z)
            joint_pos={
                "valve_joint": 0.0,
            },
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                # NOTE: Used only when we apply a specific control.
                joint_names_expr=["valve_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # NOTE: We have to use below terms through `self.cfg.~`
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # TODO
    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 3.0
    open_reward_scale = 1.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


def _get_pose_b(obj_w, base_w):
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        base_w[:, 0:3], base_w[:, 3:7], obj_w[:, 0:3], obj_w[:, 3:7]
    )
    return obj_pos_b, obj_quat_b

def matrix_from_pose(trans, quat):
    n_envs = trans.shape[0]
    T = SE3.Alloc(n_envs)
    trans, quat = trans.cpu().numpy(), quat.cpu().numpy()
    for i in range(n_envs):
        T[i] = SE3(trans[i]) * UnitQuaternion(quat[i]).SE3()
    return T

def make_traj(T_init, T_des, step):
    n_envs = len(T_init)
    trajs = torch.zeros(n_envs, step, 7)
    for i in range(n_envs):
        trajs[i] = torch.cat([torch.cat([torch.from_numpy(traj[:3, 3]).unsqueeze(0),
                                         quat_from_matrix(torch.from_numpy(traj[:3, :3])).unsqueeze(0)], dim=-1)
                              for traj in ctraj(T_init[i], T_des[i], step).data])
    return trajs


class FrankaValveGraspEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaValveGraspEnvCfg

    def __init__(self, cfg: FrankaValveGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        # NOTE: 'local_pose' is a fixed frame in the environment.
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        # NOTE: 'finger_pose' is a transformation from "world" to "finger"
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0  # NOTE: Isn't it useless?
        finger_pose[3:7] = lfinger_pose[3:7]
        # NOTE: 'hand_pose_inv' is a transformation from "hand" to "world"
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        # NOTE: "hand" to "world" + "world" to "finger" => "hand" to "finger"
        robot_local_grasp_pose_rot, robot_local_grasp_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_grasp_pose_pos += torch.tensor([0, 0, 0.04], device=self.device)
        self.robot_local_grasp_pos = robot_local_grasp_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        valve_local_center_pose = torch.tensor([0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.valve_local_center_pos = valve_local_center_pose[0:3].repeat((self.num_envs, 1))
        self.valve_local_center_rot = valve_local_center_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.valve_inward_axis = torch.tensor([0, 0, -1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs, 1)
        # )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.valve_center_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.valve_center_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.valve_init_state = torch.zeros((self.num_envs), device=self.device)

        # For IK controller

        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)
        # self.cmd_limit = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device)
        self.cmd_limit = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)
        self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.device)

        self.base_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pos_b = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self.ee_quat_b = torch.zeros(self.scene.num_envs, 4, device=self.device)
        self.valve_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.valve_pos_b = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self.valve_quat_b = torch.zeros(self.scene.num_envs, 4, device=self.device)

        self.jacobian = torch.zeros(self.scene.num_envs, 6, 7, device=self.device)
        self.joint_pos = torch.zeros(self.scene.num_envs, 7, device=self.device)

        self.traj_cnt = 0
        self.traj_t = 300
        self.trajs = torch.zeros(self.scene.num_envs, self.traj_t, 7, device=self.device)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._valve = Articulation(self.cfg.valve)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["valve"] = self._valve

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        """ Pre-process actions before stepping through the physics.
            It is called before the physics stepping (which is decimated).  """
        # self.actions = actions.clone()
        # d_arm, d_gripper = self.actions[:, :-1], self.actions[:, -1]  # dpose
        # d_arm *= self.cmd_limit
        arm_targets = self._compute_osc_torques()
        # self.gripper_targets = torch.where(d_gripper > 0, 1, -1).unsqueeze(1).repeat(1,2)
        self.gripper_targets = torch.zeros(self.num_envs, 1).to(self.device).repeat(1,2)
        targets = torch.concat([arm_targets, self.gripper_targets], -1)
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _compute_osc_torques(self):
        # # Only needed when ``use_relative_mode=False`` in ``DifferentialIKControllerCfg``
        # d_pos, d_angle = dpose[:, :3], dpose[:, 3:]
        # d_quat = quat_from_euler_xyz(*d_angle.T)
        # dpose = torch.concat([d_pos, d_quat], dim=-1)
        # compute the joint commands
        self.diff_ik_controller.reset()
        # self.diff_ik_controller.set_command(dpose, ee_pos_b, ee_quat_b)
        self.diff_ik_controller.set_command(self.trajs[:, self.traj_cnt, :])
        joint_pos_des = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, self.jacobian, self.joint_pos)
        if self.traj_cnt < self.traj_t - 1:
            self.traj_cnt += 1

        return joint_pos_des

    def _apply_action(self):
        """ Apply actions to the simulator.
            It is called at each physics time-step.

            This function does not apply the joint targets to the simulation.
            It only fills the buffers with the desired values.
            To apply the joint targets, call the write_data_to_sim() function.  """
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.valve_init_state - self._valve.data.joint_pos.squeeze(-1) > 1.57  # TODO
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:  # TODO
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pose = self._robot.data.body_state_w[:, self.left_finger_link_idx][:, :7]
        robot_right_finger_pose = self._robot.data.body_state_w[:, self.right_finger_link_idx][:, :7]

        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,
            self.valve_init_state,
            self.robot_grasp_pos,
            self.robot_grasp_rot,
            self.valve_center_pos,
            self.valve_center_rot,
            robot_left_finger_pose,
            robot_right_finger_pose,
            self.gripper_forward_axis,
            self.valve_inward_axis,
            self.gripper_up_axis,
            # self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.valve_init_state[env_ids] = self._valve.data.joint_pos.squeeze(-1)[env_ids]

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        # Trajectory planning for each environment
        self.traj_cnt = 0
        ee_mat = matrix_from_pose(self.ee_pos_b, self.ee_quat_b)
        des_mat = matrix_from_pose(self.valve_pos_b, self.valve_quat_b)
        self.trajs = make_traj(ee_mat, des_mat, self.traj_t).to(self.device)


    def _get_observations(self) -> dict:
        ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        base_pose_w = self._robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = _get_pose_b(ee_pose_w, base_pose_w)
        # TODO: Consider the valve perception through camera
        valve_pose_w = torch.cat([self.valve_center_pos, self.valve_center_rot], dim=-1)
        valve_pos_b, valve_quat_b = _get_pose_b(valve_pose_w, base_pose_w)

        obs = torch.cat(
            (
                ee_pos_b,                                           # 3
                torch.stack(euler_xyz_from_quat(ee_quat_b)).T,      # 3
                valve_pos_b,                                        # 3
                torch.stack(euler_xyz_from_quat(valve_quat_b)).T,   # 3
                self._robot.data.joint_pos[:, -2:]                  # 2
            ),
            dim=-1,
        )
        """ A dictionary should be returned that contains policy as the key,
            and the full observation buffer as the value.
            For asymmetric policies,
            the dictionary should also include the key critic and the states buffer as the value.   """
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        # NOTE: Following outputs are transformations from "world" frame.
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.valve_center_rot,
            self.valve_center_pos,  # TODO
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

        self.jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        self.joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # For trajectory planning
        self.base_pose_w = self._robot.data.root_state_w[:, 0:7]
        self.ee_pose_w = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.ee_pos_b, self.ee_quat_b = _get_pose_b(self.ee_pose_w, self.base_pose_w)
        self.valve_pose_w = self._valve.data.body_state_w[:, self.valve_link_idx, :7]
        self.valve_pos_b, self.valve_quat_b = _get_pose_b(self.valve_pose_w, self.base_pose_w)

    def _compute_rewards(  # TODO
        self,
        actions,
        valve_dof_pos,
        valve_init_state,
        robot_grasp_pos,
        robot_grasp_rot,
        valve_center_pos,
        valve_center_rot,
        robot_lfinger_pose,
        robot_rfinger_pose,
        gripper_forward_axis,
        valve_inward_axis,
        gripper_up_axis,
        # drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the valve in z axis
        robot_grasp_pos_v, _ = _get_pose_b(
            torch.cat((robot_grasp_pos, robot_grasp_rot), dim=-1),
            torch.cat((valve_center_pos, valve_center_rot), dim=-1)
        )
        d_xy = torch.norm(robot_grasp_pos_v[:, :-1], p=2, dim=-1) - 0.11  # radius = 0.11
        d_z = torch.abs(robot_grasp_pos_v[:, -1]) * 5
        d = d_xy ** 2 + d_z ** 2
        # d = torch.norm(robot_grasp_pos_v[:, -1], p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d)
        # dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(robot_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(valve_center_rot, valve_inward_axis)
        # axis3 = tf_vector(robot_grasp_rot, gripper_up_axis)  # TODO: Make gripper_up_axis normal to valve
        # axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        # dot2 = (
        #     torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 1.0 * (torch.sign(dot1) * dot1**2)
        # rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the valve has been rotated
        open_reward = valve_init_state - valve_dof_pos.squeeze(-1)

        # # penalty for distance of each finger from the drawer handle
        robot_lfinger_pos_v, _ = _get_pose_b(
            robot_lfinger_pose,
            torch.cat((valve_center_pos, valve_center_rot), dim=-1)
        )
        robot_rfinger_pos_v, _ = _get_pose_b(
            robot_rfinger_pose,
            torch.cat((valve_center_pos, valve_center_rot), dim=-1)
        )
        lfinger_dist = torch.norm(robot_lfinger_pos_v[:, :2], p=2, dim=-1) - 0.11
        rfinger_dist = 0.11 - torch.norm(robot_rfinger_pos_v[:, :2], p=2, dim=-1)
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_dist": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_dist": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(valve_dof_pos.squeeze(-1) > 0.1, rewards + 0.25, rewards)
        rewards = torch.where(valve_dof_pos.squeeze(-1) > 0.5, rewards + 0.25, rewards)
        rewards = torch.where(valve_dof_pos.squeeze(-1) > 1.5, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        robot_local_grasp_rot,
        robot_local_grasp_pos,
    ):
        # NOTE: 'hand_pose' = "world" to "hand", 'robot_local_grasp_pose' = "hand" to "finger"
        # NOTE: So we can make "world" to "finger" transformation using `tf_combine` function.
        robot_global_rot, robot_global_pos = tf_combine(
            hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        )
        # valve_global_rot, valve_global_pos = tf_combine(  # TODO
        #     hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos
        # )

        return (robot_global_rot,
                robot_global_pos,
                self._valve.data.body_quat_w[:, self.valve_link_idx],
                self._valve.data.body_pos_w[:, self.valve_link_idx])
