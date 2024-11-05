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

@configclass
class EventCfg:
    # robot
    franka_joint_odd = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1357]"),
            "position_range": (-0.5, 0.5),
            "velocity_range": (0, 0)
        }
    )
    franka_joint_even = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[246]"),
            "position_range": (-0.3, 0.3),
            "velocity_range": (0, 0)
        }
    )

    # valve
    valve_joint = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve", joint_names=".*"),
            "position_range": (-0.5, 0.5),
            "velocity_range": (0, 0)
        }
    )
    valve_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("valve", body_names=".*"),
            "pose_range": {"x": (-0.1, 0.1),
                           "y": (-0.1, 0.1),
                           "z": (-0.1, 0.1),
                           "roll": (-0.349, 0.349),
                           "pitch": (0, 1.57),  # 45 deg
                           "yaw": (-0.349, 0.349),  # 20 deg
                           },
            "velocity_range": {}
        }
    )


@configclass
class FrankaValveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
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
                # "panda_joint1": 1.157,
                # "panda_joint2": -1.066,
                # "panda_joint3": -0.155,
                # "panda_joint4": -2.239,
                # "panda_joint5": -1.841,
                # "panda_joint6": 1.003,
                # "panda_joint7": 0.469,
                # "panda_finger_joint.*": 0.035,
                "panda_joint1": 0,
                "panda_joint2": 0,
                "panda_joint3": 0,
                "panda_joint4": (-0.0698 - 3.0718) / 2,  # -1.5708
                "panda_joint5": 0,
                "panda_joint6": (3.7525 - 0.0175) / 2,  # 1.8675
                "panda_joint7": 0,
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
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0


class FrankaValveEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaValveEnvCfg

    def __init__(self, cfg: FrankaValveEnvCfg, render_mode: str | None = None, **kwargs):
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
        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        # drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        # self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        # self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs, 1)
        # )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        # self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
        #     (self.num_envs, 1)
        # )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        # self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

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
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminated = self._cabinet.data.joint_pos[:, 3] > 0.39
        # truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards(
            self.actions,
            self.cfg.action_penalty_scale,
        )

    # def _reset_idx(self, env_ids: torch.Tensor | None):
    #     super()._reset_idx(env_ids)
    #
    #     # robot state
    #     joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
    #         -0.125,
    #         0.125,
    #         (len(env_ids), self._robot.num_joints),
    #         self.device,
    #     )
    #     joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
    #     joint_vel = torch.zeros_like(joint_pos)
    #     self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    #
    #     # valve state
    #     # # |-- root pose
    #     # valve_pos = self._valve.data.body_state_w[env_ids][:, 0, :3]
    #     # # + sample_uniform(
    #     # #     -0.1,
    #     # #     0.1,
    #     # #     (len(env_ids), 3),
    #     # #     self.device,
    #     # # )
    #     # valve_quat = self._valve.data.body_state_w[env_ids][:, 0, 3:7] + sample_uniform(
    #     #     -0.125,
    #     #     0.125,
    #     #     (len(env_ids), 4),
    #     #     self.device,
    #     # )
    #     # self._valve.write_root_pose_to_sim(torch.cat((valve_pos, valve_quat), 1))
    #     # |-- joint
    #     valve_joint_pos = self._valve.data.default_joint_pos[env_ids] + sample_uniform(
    #         -1.57,
    #         1.57,
    #         (len(env_ids), self._valve.num_joints),
    #         self.device,
    #     )
    #     valve_joint_vel = torch.zeros((len(env_ids), self._valve.num_joints), device=self.device)
    #     self._valve.write_joint_state_to_sim(valve_joint_pos, valve_joint_vel, env_ids=env_ids)
    #
    #
    #     # Need to refresh the intermediate values so that _get_observations() can use the latest values
    #     # self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        # to_target = self.drawer_grasp_pos - self.robot_grasp_pos

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                # to_target,
                # self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                # self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        # drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        # drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        # # NOTE: Following outputs are transformations from "world" frame.
        # (
        #     self.robot_grasp_rot[env_ids],
        #     self.robot_grasp_pos[env_ids],
        #     self.drawer_grasp_rot[env_ids],
        #     self.drawer_grasp_pos[env_ids],
        # ) = self._compute_grasp_transforms(
        #     hand_rot,
        #     hand_pos,
        #     self.robot_local_grasp_rot[env_ids],
        #     self.robot_local_grasp_pos[env_ids],
        #     drawer_rot,
        #     drawer_pos,
        #     self.drawer_local_grasp_rot[env_ids],
        #     self.drawer_local_grasp_pos[env_ids],
        # )

    def _compute_rewards(
        self,
        actions,
        action_penalty_scale,
    ):
        # # distance from hand to the drawer
        # d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d**2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)
        #
        # axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        # axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        # axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        # axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)
        #
        # dot1 = (
        #     torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of forward axis for gripper
        # dot2 = (
        #     torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        # )  # alignment of up axis for gripper
        # # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        # rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # # how far the cabinet has been opened out
        # open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint
        #
        # # penalty for distance of each finger from the drawer handle
        # lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        # rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        # finger_dist_penalty = torch.zeros_like(lfinger_dist)
        # finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        # finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            # dist_reward_scale * dist_reward
            # + rot_reward_scale * rot_reward
            # + open_reward_scale * open_reward
            # + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            # "dist_reward": (dist_reward_scale * dist_reward).mean(),
            # "rot_reward": (rot_reward_scale * rot_reward).mean(),
            # "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            # "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            # "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            # "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # # bonus for opening drawer properly
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        # rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    # def _compute_grasp_transforms(
    #     self,
    #     hand_rot,
    #     hand_pos,
    #     franka_local_grasp_rot,
    #     franka_local_grasp_pos,
    #     drawer_rot,
    #     drawer_pos,
    #     drawer_local_grasp_rot,
    #     drawer_local_grasp_pos,
    # ):
    #     # NOTE: 'hand_pose' = "world" to "hand", 'franka_local_grasp_pose' = "hand" to "finger"
    #     # NOTE: So we can make "world" to "finger" transformation using `tf_combine` function.
    #     global_franka_rot, global_franka_pos = tf_combine(
    #         hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
    #     )
    #     global_drawer_rot, global_drawer_pos = tf_combine(
    #         drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
    #     )
    #
    #     return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
