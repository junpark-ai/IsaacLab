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
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, matrix_from_quat, apply_delta_pose, combine_frame_transforms, quat_from_euler_xyz, quat_rotate_inverse

@configclass
class EventCfg:
    # robot
    franka_joint_odd = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[135]"),
            "position_range": (0, 0),
            # "position_range": (-0.5, 0.5),
            "velocity_range": (0, 0)
        }
    )
    franka_joint_even = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[246]"),
            "position_range": (0, 0),
            # "position_range": (-0.5, 0.5),
            "velocity_range": (0, 0)
        }
    )
    franka_joint_7 = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint7"),
            "position_range": (0, 0),
            # "position_range": (-0.785 - 0.1, -0.785 + 0.1),
            "velocity_range": (0, 0)
        }
    )

    # valve
    valve_joint = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve", joint_names=".*"),
            # "position_range": (0, 0),
            "position_range": (-0.5, 0.5),
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
class FrankaValveReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0  # 360 timesteps
    decimation = 2
    action_space = 6  # delta pose (6)
    observation_space = 41  # ee pose (7) + valve grasp/center pose (14) + robot joint pos/vel (14) + previous action (6)
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
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # "panda_joint1": 1.157,
                # "panda_joint2": -1.066,
                # "panda_joint3": -0.155,
                # "panda_joint4": -2.239,
                # "panda_joint5": -1.841,
                # "panda_joint6": 1.003,
                # "panda_joint7": 0.469,
                # "panda_finger_joint.*": 0.035,
                "panda_joint1": 0,
                "panda_joint2": -1.066,
                "panda_joint3": 0,
                "panda_joint4": -1.57,
                "panda_joint5": 0,
                "panda_joint6": 0.785,
                "panda_joint7": -0.785,
                "panda_finger_joint.*": 0.035,
            },
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
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
            usd_path=f"/home/kist-robot2/Desktop/round_valve_experiments/collision_experiments/round_only/round_only.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0, 0.4),
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
    rot_reward_scale = 0.05
    open_penalty_scale = 0.4
    action_penalty_scale = 0.01
    finger_reward_scale = 2.0

    valve_radius = 0.12


def _get_pose_b(obj_w, base_w):
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        base_w[:, 0:3], base_w[:, 3:7], obj_w[:, 0:3], obj_w[:, 3:7]
    )
    return obj_pos_b, obj_quat_b

def is_in_forbidden_range(theta, forbidden_ranges):
    for start, end in forbidden_ranges:
        if start <= theta <= end:
            return True
    return False


class FrankaValveReachEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaValveReachEnvCfg

    def __init__(self, cfg: FrankaValveReachEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.gripper_roll_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
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
        self.valve_grasp_pose_v = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_grasp_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_grasp_pose_b = torch.zeros((self.num_envs, 7), device=self.device)

        self.valve_init_state = torch.zeros((self.num_envs, 1), device=self.device)

        # For OSC controller
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # For IK controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)
        self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.device)

        # self.cmd_limit = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device)
        self.cmd_limit = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], device=self.device)
        # self.cmd_limit = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)

        # TODO: Need to apply following variables
        self.base_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pos_b = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self.ee_quat_b = torch.zeros(self.scene.num_envs, 4, device=self.device)
        self.valve_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.valve_pos_b = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self.valve_quat_b = torch.zeros(self.scene.num_envs, 4, device=self.device)

        self.jacobian = torch.zeros(self.scene.num_envs, 6, 7, device=self.device)
        self.joint_pos = torch.zeros(self.scene.num_envs, 7, device=self.device)


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

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.valve_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/valve_center"))

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        """ Pre-process actions before stepping through the physics.
            It is called before the physics stepping (which is decimated).  """
        # # OSC
        # self.actions = actions.clone()
        # d_arm = self.actions[:]
        # # d_arm, d_gripper = self.actions[:, :-1], self.actions[:, -1]  # dpose
        # d_arm *= self.cmd_limit
        # torque_targets = self._compute_osc_torques(d_arm)
        # self.gripper_targets = torch.where(d_arm[:, -1] > 0, 1, 1).unsqueeze(1).repeat(1, 2)
        # # pos_targets = torch.concat([arm_targets, self.gripper_targets], -1)
        # self.robot_pos_targets = torch.clamp(
        #     self.gripper_targets,
        #     self.robot_dof_lower_limits[-2:], self.robot_dof_upper_limits[-2:]
        # )
        # self.robot_torque_targets = torch.clamp(
        #     torque_targets,
        #     -torch.tensor([87] * 4 + [12] * 3).to(self.device), torch.tensor([87] * 4 + [12] * 3).to(self.device)
        # )
        # # self.robot_torque_targets[:, :-2] = torque_targets

        # IK
        self.actions = actions.clone()
        d_arm, d_gripper = self.actions[:], self.actions[:, -1]  # dpose
        d_arm *= self.cmd_limit
        arm_targets = self._compute_osc_torques(d_arm)
        self.gripper_targets = torch.where(d_gripper > 0, 1, 1).unsqueeze(1).repeat(1,2)
        targets = torch.concat([arm_targets, self.gripper_targets], -1)
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _compute_osc_torques(self, dpose):
        # OSC
        # q = self._robot.data.joint_pos[:, :7]
        # qd = self._robot.data.joint_vel[:, :7]
        # j_eef = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        # mm = self._robot.root_physx_view.get_mass_matrices()[:, :7, :7]
        # mm_inv = torch.inverse(mm)
        # m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        # m_eef = torch.inverse(m_eef_inv)
        #
        # # Transform our cartesian action `dpose` into joint torques `u`
        # kp = torch.tensor([150.] * 6, device=self.device)
        # kd = 2 * torch.sqrt(kp)
        # kp_null = torch.tensor([0.] * 7, device=self.device)
        # kd_null = 2 * torch.sqrt(kp_null)
        # ee_vel_w = self._robot.root_physx_view.get_link_velocities()[:, self.ee_jacobi_idx + 1, :]
        # # base_rotm_w = matrix_from_quat(self._robot.data.root_state_w[:, 3:7])
        # base_rotm_w = matrix_from_quat(self._robot.data.body_quat_w[:, self.ee_jacobi_idx + 1, :])
        # ee_vel_lin_b = (base_rotm_w @ ee_vel_w[:, :3].unsqueeze(-1)).squeeze(-1)
        # ee_vel_ang_b = (base_rotm_w @ ee_vel_w[:, 3:].unsqueeze(-1)).squeeze(-1)
        # ee_vel_b = torch.cat((ee_vel_lin_b, ee_vel_ang_b), dim=-1)
        # u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        #         kp * dpose - kd * ee_vel_b).unsqueeze(-1)
        #
        # # Nullspace control torques `u_null` prevents large changes in joint configuration
        # # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # # roboticsproceedings.org/rss07/p31.pdf
        # j_eef_inv = m_eef @ j_eef @ mm_inv
        # u_null = kd_null * -qd + kp_null * (
        #         (self._robot.data.default_joint_pos[0, :7] - q + torch.pi) % (2 * torch.pi) - torch.pi)
        # u_null[:, 7:] *= 0
        # u_null = mm @ u_null.unsqueeze(-1)
        # u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
        #
        # return u.squeeze(-1)

        # IK
        # compute the joint commands
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(dpose, self.ee_pos_b, self.ee_quat_b)
        joint_pos_des = self.diff_ik_controller.compute(self.ee_pos_b, self.ee_quat_b, self.jacobian, self.joint_pos)
        return joint_pos_des


    def _apply_action(self):
        """ Apply actions to the simulator.
            It is called at each physics time-step.

            This function does not apply the joint targets to the simulation.
            It only fills the buffers with the desired values.
            To apply the joint targets, call the write_data_to_sim() function.  """
        # OSC
        # self._robot.set_joint_effort_target(target=self.robot_torque_targets, joint_ids=[i for i in range(7)])
        # self._robot.set_joint_position_target(target=self.robot_pos_targets, joint_ids=[7, 8])

        # IK
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminated = self.valve_init_state - self._valve.data.joint_pos.squeeze(-1) > 1.57  # TODO
        terminated = False
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
            self.valve_grasp_pose_w,
            robot_left_finger_pose,
            robot_right_finger_pose,
            self.gripper_forward_axis,
            self.valve_inward_axis,
            self.gripper_roll_axis,
            # self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_penalty_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
            self.ee_pos_b,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.valve_init_state[env_ids] = self._valve.data.joint_pos[env_ids]

        # Define the forbidden ranges in radians
        forbidden_ranges = [
            (torch.tensor(0.0), torch.tensor(10.0) * torch.pi / 180),
            (torch.tensor(80.0) * torch.pi / 180, torch.tensor(100.0) * torch.pi / 180),
            (torch.tensor(170.0) * torch.pi / 180, torch.tensor(190.0) * torch.pi / 180),
            (torch.tensor(260.0) * torch.pi / 180, torch.tensor(280.0) * torch.pi / 180),
            (torch.tensor(350.0) * torch.pi / 180, torch.tensor(360.0) * torch.pi / 180)
        ]

        # Sample grasp_theta
        grasp_theta = 2 * torch.pi * torch.rand(len(env_ids)).to(self.device)

        # Check which samples are invalid (fall into forbidden ranges)
        invalid_mask = torch.tensor([is_in_forbidden_range(g, forbidden_ranges) for g in grasp_theta])

        # Re-sample only invalid values
        while invalid_mask.any():
            # Re-sample the invalid ones
            grasp_theta[invalid_mask] = 2 * torch.pi * torch.rand(invalid_mask.sum()).to(self.device)

            # Re-check the invalid values
            invalid_mask = torch.tensor([is_in_forbidden_range(g, forbidden_ranges) for g in grasp_theta])

        self.valve_grasp_pose_v[env_ids, 0] = torch.cos(grasp_theta) * self.cfg.valve_radius
        self.valve_grasp_pose_v[env_ids, 1] = torch.sin(grasp_theta) * self.cfg.valve_radius
        self.valve_grasp_pose_v[env_ids, 3:] = quat_from_euler_xyz(torch.zeros_like(grasp_theta), torch.pi * torch.ones_like(grasp_theta), grasp_theta - torch.pi / 2)
        # self.valve_grasp_pose_v[env_ids, 3:] = quat_from_euler_xyz(torch.zeros_like(grasp_theta), torch.zeros_like(grasp_theta), torch.zeros_like(grasp_theta))

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self.init = 1  # TODO
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # OSC
        # ee_pos_b, ee_quat_b = _get_pose_b(torch.cat((self.robot_grasp_pos, self.robot_grasp_rot), dim=-1), self.base_pose_w)
        # # TODO: Consider the valve perception through camera
        # valve_grasp_pos_b, valve_grasp_quat_b = _get_pose_b(self.valve_grasp_pose_w, self.base_pose_w)
        # valve_center_pos_b, valve_center_quat_b = _get_pose_b(torch.cat((self.valve_center_pos, self.valve_center_rot), dim=-1), self.base_pose_w)
        #
        # ee_vel = self._robot.data.body_vel_w[:, self._robot.data.body_names.index('panda_hand'), :3]
        #
        # obs = torch.cat(
        #     (
        #         ee_pos_b,                                                   # 3
        #         ee_quat_b,                                                  # 4
        #         # torch.stack(euler_xyz_from_quat(ee_quat_b)).T,              # 3
        #         valve_grasp_pos_b,                                          # 3
        #         valve_grasp_quat_b,                                         # 4
        #         # torch.stack(euler_xyz_from_quat(valve_grasp_quat_b)).T,     # 3
        #         # valve_center_pos_b,                                         # 3
        #
        #         quat_rotate_inverse(self.base_pose_w[:, 3:], ee_vel)        # 3
        #
        #         # self._robot.data.joint_pos[:, -2:]                  # 2
        #     ),
        #     dim=-1,
        # )

        # IK
        ee_pos_b, ee_quat_b = _get_pose_b(torch.cat((self.robot_grasp_pos, self.robot_grasp_rot), dim=-1), self.base_pose_w)
        # TODO: Consider the valve perception through camera

        obs = torch.cat(
            (
                ee_pos_b,                                                               # 3
                ee_quat_b,                                                              # 4
                self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids],         # 7
                self._robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids],         # 7
                self.valve_grasp_pose_b,                                                # 7
                self.actions,                                                           # 6
                self.valve_center_pos_b,                                                # 3
                self.valve_center_quat_b                                                # 4
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
            self.valve_center_rot[env_ids],
            self.valve_center_pos[env_ids],  # TODO
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

        # TODO: Make other variables in the `_compute_osc_torques` function like this
        self.jacobian[:] = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        self.joint_pos[:] = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # For trajectory planning
        self.base_pose_w[:] = self._robot.data.root_state_w[:, 0:7]
        self.ee_pose_w[:] = self._robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.ee_pos_b[:], self.ee_quat_b[:] = _get_pose_b(self.ee_pose_w, self.base_pose_w)
        self.valve_pose_w[:] = self._valve.data.body_state_w[:, self.valve_link_idx, :7]
        # grasp_pos_w = tf_vector(self.valve_center_rot, self.valve_grasp_pos[env_ids, :])
        # self.valve_pose_w[:, :3] += grasp_pos_w  # TODO

        self.valve_grasp_pose_w[:] = torch.cat(
            combine_frame_transforms(
                self.valve_pose_w[:, :3], self.valve_pose_w[:, 3:],
                self.valve_grasp_pose_v[:, :3], self.valve_grasp_pose_v[:, 3:],
            ),
            dim=-1
        )
        # self.valve_pose_w[:] = torch.cat(
        #     apply_delta_pose(self.valve_pose_w[:, :3], self.valve_pose_w[:, 3:], self.valve_grasp_pose_v), dim=-1
        # )
        self.valve_pos_b[:], self.valve_quat_b[:] = _get_pose_b(self.valve_pose_w, self.base_pose_w)
        if self.init:
            self.valve_grasp_pose_b[:, :3], self.valve_grasp_pose_b[:, 3:] = _get_pose_b(self.valve_grasp_pose_w, self.base_pose_w)
            self.valve_center_pos_b, self.valve_center_quat_b = _get_pose_b(  # TODO: Initialize
                torch.cat((self.valve_center_pos, self.valve_center_rot), dim=-1), self.base_pose_w)
            self.init = 0

        # Markers
        self.ee_marker.visualize(self.robot_grasp_pos[:], self.robot_grasp_rot[:])
        # self.ee_marker.visualize(self.ee_pose_w[:, 0:3], self.ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.valve_grasp_pose_w[:, 0:3], self.valve_grasp_pose_w[:, 3:7])
        self.valve_marker.visualize(self.valve_center_pos, self.valve_center_rot)

    def _compute_rewards(  # TODO
        self,
        actions,
        valve_dof_pos,
        valve_init_state,
        robot_grasp_pos,
        robot_grasp_rot,
        valve_center_pos,
        valve_center_rot,
        valve_grasp_pose,
        robot_lfinger_pose,
        robot_rfinger_pose,
        gripper_forward_axis,
        valve_inward_axis,
        gripper_roll_axis,
        # drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_penalty_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
        ee_pos_b,
    ):

        d = torch.norm(robot_grasp_pos - valve_grasp_pose[:, :3], dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * d)

        axis1 = tf_vector(robot_grasp_rot, gripper_roll_axis)
        axis2 = tf_vector(valve_grasp_pose[:, 3:7], gripper_roll_axis)
        axis3 = tf_vector(robot_grasp_rot, gripper_forward_axis)
        axis4 = tf_vector(valve_grasp_pose[:, 3:7], gripper_forward_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        # rot_reward = 1.0 * (torch.sign(dot1) * dot1**2)
        rot_reward = (0.7 * torch.sign(dot1) * dot1**2
                    + 0.3 * torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the valve has been rotated
        open_penalty = torch.abs((valve_dof_pos - valve_init_state).squeeze(-1))
        # open_reward = valve_init_state - valve_dof_pos.squeeze(-1)

        # # # penalty for distance of each finger from the drawer handle
        # robot_lfinger_pos_v, _ = _get_pose_b(
        #     robot_lfinger_pose,
        #     torch.cat((valve_center_pos, valve_center_rot), dim=-1)
        # )
        # robot_rfinger_pos_v, _ = _get_pose_b(
        #     robot_rfinger_pose,
        #     torch.cat((valve_center_pos, valve_center_rot), dim=-1)
        # )
        # lfinger_dist = torch.norm(robot_lfinger_pos_v[:, :2], p=2, dim=-1) - self.cfg.valve_radius
        # rfinger_dist = self.cfg.valve_radius - torch.norm(robot_rfinger_pos_v[:, :2], p=2, dim=-1)
        # finger_dist_penalty = torch.zeros_like(lfinger_dist)
        # finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        # finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            - open_penalty_scale * open_penalty
            # + finger_reward_scale * finger_dist_penalty
            # - action_penalty_scale * action_penalty
        )

        # rewards = torch.where(ee_pos_b[:, 0] < 0, rewards - 0.25, rewards)

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_penalty": (-open_penalty_scale * open_penalty).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            # "left_finger_dist": (finger_reward_scale * lfinger_dist).mean(),
            # "right_finger_dist": (finger_reward_scale * rfinger_dist).mean(),
            # "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

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
