# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents
from .franka_cabinet_env import FrankaCabinetEnv, FrankaCabinetEnvCfg
from .franka_cabinet_ik_env import FrankaCabinetIKEnv, FrankaCabinetIKEnvCfg
from .franka_cabinet_osc_env import FrankaCabinetOSCEnv, FrankaCabinetOSCEnvCfg
from .franka_valve_env import FrankaValveEnv, FrankaValveEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_cabinet:FrankaCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCabinetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v1",  # IK controller
    entry_point="omni.isaac.lab_tasks.direct.franka_cabinet:FrankaCabinetIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCabinetIKEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:IK_rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v2",  # OSC controller
    entry_point="omni.isaac.lab_tasks.direct.franka_cabinet:FrankaCabinetOSCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCabinetOSCEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:OSC_rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Valve-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.franka_cabinet:FrankaValveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaValveEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
