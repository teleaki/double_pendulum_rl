from isaaclab.app import AppLauncher


# Start Isaac Sim with a GUI. Isaac Lab modules that depend on Isaac Sim must
# be imported after the application has started.
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import carb
import omni.appwindow

# Importing this package executes env/__init__.py and registers the Gym task.
import double_pendulum_rl.env  # noqa: F401
from double_pendulum_rl.env.balance_env import BalanceEnvCfg


TASK_NAME = "WheelLeg-Balance-Direct-v0"
NUM_TEST_STEPS = 1200
ACTION_NOISE_SCALE = 0.05
PRINT_INTERVAL = 60


def main():
    """Run the balance environment with small random actions."""

    # Keep this test deterministic so repeated runs are easier to compare.
    torch.manual_seed(42)

    env_cfg = BalanceEnvCfg()
    env_cfg.seed = 42

    # Fabric cloning is fast, but custom URDF assets can show incomplete
    # cloned geometry in the GUI. Disable it for this visual smoke test.
    env_cfg.scene.clone_in_fabric = False

    env = gym.make(
        TASK_NAME,
        cfg=env_cfg,
        render_mode="human",
    )
    base_env = env.unwrapped

    # Show the complete 4x4 environment grid.
    base_env.sim.set_camera_view(
        eye=(7.0, 7.0, 5.0),
        target=(0.0, 0.0, 0.4),
    )

    # The keyboard callback only records a request. The actual reset is done
    # in the simulation loop so physics state is not changed inside a UI
    # callback.
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

    try:
        observations, _ = env.reset()
        policy_observations = observations["policy"]

        print("[INFO] Balance environment created.")
        print(f"[INFO] Number of environments: {base_env.num_envs}")
        print(f"[INFO] Action shape: ({base_env.num_envs}, {env_cfg.action_space})")
        print(f"[INFO] Observation shape: {tuple(policy_observations.shape)}")
        print(f"[INFO] Device: {base_env.device}")
        print(
            f"[INFO] Running {NUM_TEST_STEPS} policy steps with "
            f"random-action scale {ACTION_NOISE_SCALE}."
        )
        print("[INFO] Falling and automatic resets are expected in this test.")
        print("[INFO] Press R to reset all environments.")

        if policy_observations.shape != (
            base_env.num_envs,
            env_cfg.observation_space,
        ):
            raise RuntimeError(
                "Unexpected observation shape: "
                f"{tuple(policy_observations.shape)}"
            )

        total_terminated = 0
        total_time_outs = 0

        for step in range(1, NUM_TEST_STEPS + 1):
            if not simulation_app.is_running():
                break

            if reset_requested[0]:
                observations, _ = env.reset()
                policy_observations = observations["policy"]
                reset_requested[0] = False
                print("[INFO] Manual reset: all environments were reset.")

            actions = ACTION_NOISE_SCALE * torch.randn(
                (base_env.num_envs, env_cfg.action_space),
                dtype=torch.float32,
                device=base_env.device,
            )
            actions.clamp_(-1.0, 1.0)

            observations, rewards, terminated, time_outs, _ = env.step(actions)
            policy_observations = observations["policy"]

            if not torch.isfinite(policy_observations).all():
                raise RuntimeError(
                    f"Non-finite observation detected at step {step}."
                )

            if not torch.isfinite(rewards).all():
                raise RuntimeError(
                    f"Non-finite reward detected at step {step}."
                )

            terminated_count = torch.count_nonzero(terminated).item()
            time_out_count = torch.count_nonzero(time_outs).item()
            total_terminated += terminated_count
            total_time_outs += time_out_count

            if step == 1 or step % PRINT_INTERVAL == 0:
                root_height = (
                    base_env.robot.data.root_pos_w[:, 2]
                    - base_env.scene.env_origins[:, 2]
                )

                print(
                    f"[STATE] step={step:4d} "
                    f"reward_mean={rewards.mean().item(): .4f} "
                    f"height_mean={root_height.mean().item(): .4f} "
                    f"terminated={terminated_count} "
                    f"time_outs={time_out_count}"
                )

        print("[PASS] Environment smoke test finished without NaN or Inf.")
        print(f"[INFO] Total terminated resets: {total_terminated}")
        print(f"[INFO] Total time-out resets: {total_time_outs}")

    finally:
        input_interface.unsubscribe_to_keyboard_events(
            keyboard,
            keyboard_subscription,
        )
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
