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
from omni.isaac.lab.utils.math import (subtract_frame_transforms, matrix_from_quat, combine_frame_transforms,
                                       quat_from_euler_xyz, quat_rotate_inverse, skew_symmetric_matrix)

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


# controller = "OSC"
controller = "IK"

@configclass
class FrankaValveReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0  # 360 timesteps
    decimation = 2
    if controller == "OSC":
        # OSC
        action_space = 6  # delta pose (6)  # TODO
        observation_space = 21  # ee pose (7) + valve grasp/center pose (14)
    if controller == "IK":
        # IK
        action_space = 6  # delta pose (6)  # TODO
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
                stiffness=400.0,  # TODO
                damping=40.0,  # TODO
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,  # TODO
                damping=40.0,  # TODO
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
            rot=(1, 0, 0, 0),
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

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 0.05
    open_penalty_scale = 0.4
    action_penalty_scale = 0.01

    valve_radius = 0.12


def _get_pose_b(obj_w, base_w):
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        base_w[:, :3], base_w[:, 3:], obj_w[:, :3], obj_w[:, 3:]
    )
    obj_pose_b = torch.cat((obj_pos_b, obj_quat_b), dim=-1)
    return obj_pose_b

def _combine_pose(pose1, pose2):
    combined_pos, combined_quat = combine_frame_transforms(
        pose1[:, :3], pose1[:, 3:], pose2[:, :3], pose2[:, 3:]
    )
    combined_pose = torch.cat((combined_pos, combined_quat), dim=-1)
    return combined_pose

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

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.hand2ee_pose = torch.tensor([0, 0, 0.107, 1, 0, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_roll_axis = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_hand")[0][0]
        self.valve_link_idx = self._valve.find_bodies("valve")[0][0]

        self.valve_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.valve_pose_b = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.valve_center_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_center_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_grasp_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_grasp_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_grasp_pose_v = torch.zeros((self.num_envs, 7), device=self.device)
        self.valve_init_state = torch.zeros((self.num_envs, 1), device=self.device)

        self.base_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pose_w = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_pose_b = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.ee_vel_w = torch.zeros(self.scene.num_envs, 6, device=self.device)

        self.jacobian = torch.zeros(self.scene.num_envs, 6, 7, device=self.device)
        self.joint_pos = torch.zeros(self.scene.num_envs, 7, device=self.device)
        self.actions = torch.zeros(self.scene.num_envs, self.cfg.action_space, device=self.device)

        # For OSC controller
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        if self._robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]
        self.robot_pos_targets = torch.zeros(self.scene.num_envs, 2, device=self.device)
        self.robot_torque_targets = torch.zeros(self.scene.num_envs, 7, device=self.device)

        # For IK controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)
        self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.cmd_limit = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0], device=self.device)
        # self.cmd_limit = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=self.device)



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
        if controller == "OSC":
            # OSC
            self.actions[:] = actions.clone()
            d_arm, d_gripper = self.actions[:], self.actions[:, -1]  # TODO
            # d_arm, d_gripper = self.actions[:, :-1], self.actions[:, -1]
            d_arm *= self.cmd_limit
            torque_targets = self._custom_controller(d_arm)
            gripper_targets = torch.where(d_gripper > 0, 1, 1).unsqueeze(1).repeat(1, 2)  # TODO
            # gripper_targets = torch.where(d_gripper > 0, 1, 0).unsqueeze(1).repeat(1, 2)
            # pos_targets = torch.concat([arm_targets, gripper_targets], -1)
            self.robot_pos_targets[:] = torch.clamp(
                gripper_targets,
                self.robot_dof_lower_limits[-2:], self.robot_dof_upper_limits[-2:]
            )
            self.robot_torque_targets[:] = torch.clamp(
                torque_targets,
                -torch.tensor([87] * 4 + [12] * 3).to(self.device), torch.tensor([87] * 4 + [12] * 3).to(self.device)
            )
        if controller == "IK":
            # IK
            self.actions[:] = actions.clone()
            d_arm, d_gripper = self.actions[:], self.actions[:, -1]  # TODO
            # d_arm, d_gripper = self.actions[:, :-1], self.actions[:, -1]
            d_arm *= self.cmd_limit
            arm_targets = self._custom_controller(d_arm)
            gripper_targets = torch.where(d_gripper > 0, 1, 1).unsqueeze(1).repeat(1, 2)  # TODO
            # gripper_targets = torch.where(d_gripper > 0, 1, 0).unsqueeze(1).repeat(1, 2)
            targets = torch.concat([arm_targets, gripper_targets], -1)
            self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)


    def _custom_controller(self, dpose):
        if controller == "OSC":
            # OSC
            q = self._robot.data.joint_pos[:, :7]
            qd = self._robot.data.joint_vel[:, :7]
            mm = self._robot.root_physx_view.get_mass_matrices()[:, :7, :7]
            mm_inv = torch.inverse(mm)
            m_eef_inv = self.jacobian @ mm_inv @ torch.transpose(self.jacobian, 1, 2)
            m_eef = torch.inverse(m_eef_inv)

            # Transform our cartesian action `dpose` into joint torques `u`
            kp = torch.tensor([150.] * 6, device=self.device)
            kd = 2 * torch.sqrt(kp)
            kp_null = torch.tensor([10.] * 7, device=self.device)
            kd_null = 2 * torch.sqrt(kp_null)
            base_rotm_w = matrix_from_quat(self._robot.data.root_state_w[:, 3:7])
            ee_vel_lin_b = (base_rotm_w @ self.ee_vel_w[:, :3].unsqueeze(-1)).squeeze(-1)
            ee_vel_ang_b = (base_rotm_w @ self.ee_vel_w[:, 3:].unsqueeze(-1)).squeeze(-1)
            ee_vel_b = torch.cat((ee_vel_lin_b, ee_vel_ang_b), dim=-1)
            u = torch.transpose(self.jacobian, 1, 2) @ m_eef @ (
                    kp * dpose - kd * ee_vel_b).unsqueeze(-1)

            # Nullspace control torques `u_null` prevents large changes in joint configuration
            # They are added into the nullspace of OSC so that the end effector orientation remains constant
            # roboticsproceedings.org/rss07/p31.pdf
            j_eef_inv = m_eef @ self.jacobian @ mm_inv
            u_null = kd_null * -qd + kp_null * (
                    (self._robot.data.default_joint_pos[0, :7] - q + torch.pi) % (2 * torch.pi) - torch.pi)
            u_null[:, 7:] *= 0
            u_null = mm @ u_null.unsqueeze(-1)
            u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self.jacobian, 1, 2) @ j_eef_inv) @ u_null

            return u.squeeze(-1)

        if controller == "IK":
            # IK
            # compute the joint commands
            self.diff_ik_controller.reset()
            self.diff_ik_controller.set_command(dpose, self.ee_pose_b[:, :3], self.ee_pose_b[:, 3:])
            joint_pos_des = self.diff_ik_controller.compute(self.ee_pose_b[:, :3], self.ee_pose_b[:, 3:], self.jacobian, self.joint_pos)
            return joint_pos_des


    def _apply_action(self):
        """ Apply actions to the simulator.
            It is called at each physics time-step.

            This function does not apply the joint targets to the simulation.
            It only fills the buffers with the desired values.
            To apply the joint targets, call the write_data_to_sim() function.  """
        if controller == "OSC":
            # OSC
            self._robot.set_joint_effort_target(target=self.robot_torque_targets, joint_ids=[i for i in range(7)])
            self._robot.set_joint_position_target(target=self.robot_pos_targets, joint_ids=[7, 8])

        if controller == "IK":
            # IK
            self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminated = self.valve_init_state - self._valve.data.joint_pos.squeeze(-1) > 1.57  # TODO
        terminated = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.actions,
            self._valve.data.joint_pos,
            self.valve_init_state,
            self.ee_pose_w,
            self.valve_grasp_pose_w,
            self.gripper_forward_axis,
            self.gripper_roll_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_penalty_scale,
            self.cfg.action_penalty_scale,
            self.ee_pose_b[:, :3],
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

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self.init = 1  # TODO
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        # Markers
        self.ee_marker.visualize(self.ee_pose_w[:, :3], self.ee_pose_w[:, 3:])
        self.goal_marker.visualize(self.valve_grasp_pose_w[:, :3], self.valve_grasp_pose_w[:, 3:])
        self.valve_marker.visualize(self.valve_center_pose_w[:, :3], self.valve_center_pose_w[:, 3:])

        if controller == "OSC":
            # OSC
            obs = torch.cat(
                (
                    self.ee_pose_b,                                                         # 7
                    self.valve_grasp_pose_b,                                                # 7
                    self.valve_center_pose_b,                                               # 7
                    # quat_rotate_inverse(self.base_pose_w[:, 3:], self.ee_vel_w[:, :3]),     # 3
                    # self._robot.data.joint_pos[:, -2:],                                     # 2
                ),
                dim=-1,
            )
        if controller == "IK":
            # IK
            obs = torch.cat(
                (
                    self.ee_pose_b,                                                         # 7
                    self.valve_grasp_pose_b,                                                # 7
                    self.valve_center_pose_b,                                               # 7
                    self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids],         # 7
                    self._robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids],         # 7
                    self.actions,                                                           # 6
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

        self.valve_center_pose_w[env_ids] = self._valve.data.body_state_w[env_ids, self.valve_link_idx, :7]

        self.jacobian[env_ids] = self._robot.root_physx_view.get_jacobians()[env_ids][:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        self.jacobian[env_ids, :3, :] += torch.bmm(skew_symmetric_matrix(self.hand2ee_pose[env_ids, :3]), self.jacobian[env_ids, 3:, :])
        self.jacobian[env_ids, 3:, :] = torch.bmm(matrix_from_quat(self.hand2ee_pose[env_ids, 3:]), self.jacobian[env_ids, 3:, :])
        self.joint_pos[env_ids] = self._robot.data.joint_pos[env_ids][:, self.robot_entity_cfg.joint_ids]

        self.base_pose_w[env_ids] = self._robot.data.root_state_w[env_ids, :7]
        hand_pose_w = self._robot.data.body_state_w[env_ids, self.hand_link_idx, :7]
        self.ee_pose_w[env_ids] = _combine_pose(hand_pose_w, self.hand2ee_pose[env_ids])
        self.ee_pose_b[env_ids] = _get_pose_b(self.ee_pose_w[env_ids], self.base_pose_w[env_ids])
        self.valve_pose_w[env_ids] = self._valve.data.body_state_w[env_ids, self.valve_link_idx, :7]
        self.valve_pose_b[env_ids] = _get_pose_b(self.valve_pose_w[env_ids], self.base_pose_w[env_ids])
        self.valve_grasp_pose_w[env_ids] = _combine_pose(self.valve_pose_w[env_ids], self.valve_grasp_pose_v[env_ids])
        
        if self.init:
            self.valve_grasp_pose_b[env_ids] = _get_pose_b(self.valve_grasp_pose_w[env_ids], self.base_pose_w[env_ids])
            self.valve_center_pose_b[env_ids] = _get_pose_b(
                self.valve_center_pose_w[env_ids], self.base_pose_w[env_ids])
            self.init = 0

        hand_vel_w = self._robot.root_physx_view.get_link_velocities()[env_ids, self.hand_link_idx, :]
        lin_vel_offset = torch.cross(
            hand_vel_w[:, 3:], tf_vector(hand_pose_w[env_ids, 3:], self.hand2ee_pose[env_ids, :3])
        )
        self.ee_vel_w[env_ids, :3] = hand_vel_w[:, :3] + lin_vel_offset[:, :3]
        self.ee_vel_w[env_ids, 3:] = hand_vel_w[:, 3:]

    def _compute_rewards(
        self,
        actions,
        valve_dof_pos,
        valve_init_state,
        ee_pose,
        valve_grasp_pose,
        gripper_forward_axis,
        gripper_roll_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_penalty_scale,
        action_penalty_scale,
        ee_pos_b,
    ):

        d = torch.norm(ee_pose[:, :3] - valve_grasp_pose[:, :3], dim=-1)
        dist_reward = 1 - torch.tanh(10.0 * d)

        axis1 = tf_vector(ee_pose[:, 3:], gripper_roll_axis)
        axis2 = tf_vector(valve_grasp_pose[:, 3:], gripper_roll_axis)
        axis3 = tf_vector(ee_pose[:, 3:], gripper_forward_axis)
        axis4 = tf_vector(valve_grasp_pose[:, 3:], gripper_forward_axis)

        dot1 = (  # alignment of roll(x) axis for gripper
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )
        dot2 = (  # alignment of forward axis for gripper
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )
        # reward for matching the orientation of the hand to the valve
        rot_reward = (0.7 * torch.sign(dot1) * dot1**2
                    + 0.3 * torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the valve has been rotated
        open_penalty = torch.abs((valve_dof_pos - valve_init_state).squeeze(-1))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            - open_penalty_scale * open_penalty
            - action_penalty_scale * action_penalty
        )

        rewards = torch.where(ee_pos_b[:, 0] < 0, rewards - 0.1, rewards)

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_penalty": (-open_penalty_scale * open_penalty).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        }

        return rewards
