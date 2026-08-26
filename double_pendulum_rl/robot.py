from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


# ------------------------------------------------------------
# Locate URDF
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "assets"
    / "urdf"
    / "folding_wheel_leg_robot.urdf"
)


# ------------------------------------------------------------
# Robot configuration
# ------------------------------------------------------------

ROBOT_CFG = ArticulationCfg(
    # Import robot from URDF.
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(URDF_PATH),
        # This is a mobile robot, so the chassis must not
        # be fixed to the world.
        fix_base=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    # Initial state when the robot is spawned/reset.
    init_state=ArticulationCfg.InitialStateCfg(
        # Hip is 0.06 m below the chassis center.
        #
        # Straight leg:
        #   upper leg  0.30
        # + lower leg  0.30
        # + wheel      0.10
        # + hip offset 0.06
        # = about 0.76 m
        #
        # Spawn slightly above this height.
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            ".*hip_joint": 0.7854,
            ".*knee_joint": -1.5708,
            ".*wheel_joint": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    # Motors / actuators.
    actuators={
        # Hip + knee joints.
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip_joint",
                ".*knee_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=12.0,
            # No internal PD control for now.
            # Later we will directly command joint torques.
            stiffness=0.0,
            damping=0.0,
        ),
        # Wheel joints.
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*wheel_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
