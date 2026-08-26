from isaaclab.app import AppLauncher


# Start Isaac Sim with a GUI.
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


# Import Isaac Lab modules after the app has started.
import carb
import omni.appwindow
import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from double_pendulum_rl.robot import ROBOT_CFG


@configclass
class JointTestSceneCfg(InteractiveSceneCfg):
    """Scene used for testing individual robot joints."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
    )
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(2.0, 2.0, 1.5),
        target=(0.0, 0.0, 0.4),
    )

    scene_cfg = JointTestSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]

    # Find both wheels by name instead of using fixed joint indices.
    wheel_ids, wheel_names = robot.find_joints(".*wheel_joint")
    leg_ids, leg_names = robot.find_joints(".*(hip|knee)_joint")

    print("[INFO] Joint order:")
    for index, name in enumerate(robot.joint_names):
        print(f"  {index}: {name}")

    print("[INFO] Testing joints:")
    print(f"  ids: {wheel_ids}")
    print(f"  names: {wheel_names}")
    print("[INFO] Locking leg joints for this diagnostic test:")
    print(f"  ids: {leg_ids}")
    print(f"  names: {leg_names}")
    print("[INFO] Applying +1.0 N.m to both wheels for 60 steps (0.5 seconds).")
    print("[INFO] Press R to reset and repeat the test.")

    reset_requested = [False]

    def on_keyboard_event(event, *args):
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == carb.input.KeyboardInput.R
        ):
            reset_requested[0] = True
        return True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard_event,
    )

    joint_efforts = torch.zeros(
        (scene.num_envs, robot.num_joints),
        device=robot.device,
    )
    initial_root_pos = torch.zeros(
        (scene.num_envs, 3),
        device=robot.device,
    )
    reset_root_pose = robot.data.default_root_state[:, :7].clone()
    reset_root_pose[:, :3] += scene.env_origins

    # The configured spawn height is about 1.6 cm above wheel contact for
    # the nominal leg pose. Lower it only in this diagnostic test.
    reset_root_pose[:, 2] -= 0.016
    constrained_root_pose = reset_root_pose.clone()

    constrained_root_velocity = torch.zeros(
        (scene.num_envs, 6),
        device=robot.device,
    )
    leg_target_pos = robot.data.default_joint_pos[:, leg_ids].clone()

    def reset_robot():
        constrained_root_pose.copy_(reset_root_pose)
        initial_root_pos.copy_(constrained_root_pose[:, :3])

        robot.write_root_pose_to_sim(constrained_root_pose)
        constrained_root_velocity.zero_()
        robot.write_root_velocity_to_sim(constrained_root_velocity)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        joint_efforts.zero_()
        robot.set_joint_effort_target(joint_efforts)
        robot.reset()

        print("[INFO] Robot reset; restarting two-wheel test.")

    reset_robot()

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if reset_requested[0]:
            reset_robot()
            reset_requested[0] = False
            step_count = 0

        # Diagnostic-only constraint: preserve horizontal motion while
        # fixing base height/orientation and suppressing angular motion.
        constrained_root_pose[:, :2] = robot.data.root_pos_w[:, :2]
        constrained_root_velocity.zero_()
        constrained_root_velocity[:, :2] = robot.data.root_lin_vel_w[:, :2]
        robot.write_root_pose_to_sim(constrained_root_pose)
        robot.write_root_velocity_to_sim(constrained_root_velocity)

        # Diagnostic-only joint lock: keep hip/knee positions fixed while
        # preserving the current wheel positions and velocities.
        constrained_joint_pos = robot.data.joint_pos.clone()
        constrained_joint_vel = robot.data.joint_vel.clone()
        constrained_joint_pos[:, leg_ids] = leg_target_pos
        constrained_joint_vel[:, leg_ids] = 0.0
        robot.write_joint_state_to_sim(
            constrained_joint_pos,
            constrained_joint_vel,
        )

        # Apply the same positive effort to both wheels for 60 physics steps.
        joint_efforts.zero_()
        if step_count < 60:
            joint_efforts[:, wheel_ids] = 1.0

        robot.set_joint_effort_target(joint_efforts)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step_count += 1

        if step_count in (1, 30, 60, 120):
            wheel_pos = robot.data.joint_pos[0, wheel_ids]
            wheel_vel = robot.data.joint_vel[0, wheel_ids]
            leg_pos = robot.data.joint_pos[0, leg_ids]
            root_delta = robot.data.root_pos_w[0] - initial_root_pos[0]

            print(f"[STATE] step={step_count}")
            print("  wheel_pos:", wheel_pos.detach().cpu().tolist())
            print("  wheel_vel:", wheel_vel.detach().cpu().tolist())
            print("  leg_pos:", leg_pos.detach().cpu().tolist())
            print("  root_delta_xyz:", root_delta.detach().cpu().tolist())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
