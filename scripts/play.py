from pathlib import Path

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 播放配置
# ------------------------------------------------------------

TASK_NAME = "WheelLeg-Move-Direct-v0"
MODEL_PATH = Path(__file__).resolve().parents[1] / "output" / "wheel_leg_move.onnx"
NUM_ENVS = 1
PRINT_INTERVAL = 60
SEED = 42
HEADLESS = False
INITIAL_COMMAND = (0.0, 0.0, 0.35)
KEYBOARD_CONTROL = True
RENDER_MODE = None if HEADLESS else "human"
CAMERA_EYE = (3.0, 3.0, 1.8)
CAMERA_TARGET = (0.0, 0.0, 0.5)

# GUI下可使用键盘更新项目统一的三维指令。
FORWARD_SPEED = 0.8
FAST_FORWARD_SPEED = 1.5
YAW_SPEED = 0.4
FAST_YAW_SPEED = 0.8
LEG_LENGTH_RATE = 0.05  # 按住U/J时每秒伸缩的米数


# Isaac Lab相关模块必须在AppLauncher之后导入。
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
import carb
import omni.appwindow

# 导入根包以注册所有Gym任务。
import double_pendulum_rl  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


def _shape_matches(actual_shape, expected_shape):
    """允许ONNX模型的batch维为固定值、None或动态符号。"""

    if len(actual_shape) != len(expected_shape):
        return False
    for actual, expected in zip(actual_shape, expected_shape):
        if isinstance(actual, int) and actual != expected:
            return False
    return True


def load_policy(model_path, num_envs, observation_size, action_size):
    """加载ONNX策略并验证它是否与所选环境匹配。"""

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    expected_input_shape = [num_envs, observation_size]
    expected_output_shape = [num_envs, action_size]

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Input: name={input_info.name}, shape={input_info.shape}")
    print(f"[INFO] Output: name={output_info.name}, shape={output_info.shape}")

    if not _shape_matches(input_info.shape, expected_input_shape):
        raise RuntimeError(
            f"Expected ONNX input compatible with {expected_input_shape}, "
            f"got {input_info.shape}. Check TASK_NAME and MODEL_PATH."
        )
    if not _shape_matches(output_info.shape, expected_output_shape):
        raise RuntimeError(
            f"Expected ONNX output compatible with {expected_output_shape}, "
            f"got {output_info.shape}. Check TASK_NAME and MODEL_PATH."
        )

    return session, input_info.name, output_info.name


def run_policy(
    session,
    input_name,
    output_name,
    observations,
    expected_observation_shape,
    expected_action_shape,
    device,
):
    """使用ONNX策略计算动作并转换回仿真设备。"""

    if tuple(observations.shape) != expected_observation_shape:
        raise RuntimeError(
            f"Expected observation shape {expected_observation_shape}, "
            f"got {tuple(observations.shape)}"
        )
    if not torch.isfinite(observations).all():
        raise RuntimeError("Observation contains NaN or Inf.")

    observations_numpy = observations.detach().cpu().numpy().astype(np.float32)
    actions_numpy = session.run(
        [output_name], {input_name: observations_numpy}
    )[0]

    if tuple(actions_numpy.shape) != expected_action_shape:
        raise RuntimeError(
            f"Expected action shape {expected_action_shape}, "
            f"got {tuple(actions_numpy.shape)}"
        )

    actions = torch.from_numpy(actions_numpy).to(
        device=device,
        dtype=torch.float32,
    )
    if not torch.isfinite(actions).all():
        raise RuntimeError("Policy output contains NaN or Inf.")
    return torch.clamp(actions, -1.0, 1.0)


def apply_command(base_env, command):
    """给命令式环境的全部并行机器人写入同一条指令。"""

    if command is None:
        return
    if len(command) != base_env.commands.shape[1]:
        raise RuntimeError(
            f"Expected {base_env.commands.shape[1]} command values, "
            f"got {len(command)}"
        )

    command_tensor = torch.tensor(
        command,
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env.commands[:] = command_tensor

    # 播放过程中保持外部给定的指令，不按训练周期重新采样。
    if hasattr(base_env, "command_time_left"):
        base_env.command_time_left.fill_(float("inf"))


class KeyboardCommandController:
    """把键盘按键状态转换为项目标准三维指令。"""

    def __init__(self, initial_command, leg_length_range):
        if initial_command is None:
            raise ValueError("Keyboard control requires an initial command.")

        self.initial_command = tuple(initial_command)
        self.command = list(self.initial_command)
        self.leg_length_min, self.leg_length_max = leg_length_range
        self.pressed_keys = set()
        self.reset_requested = False

        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        self.input_interface = carb.input.acquire_input_interface()
        self.subscription = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard,
            self._on_keyboard_event,
        )

    def _on_keyboard_event(self, event, *args):
        controlled_keys = {
            carb.input.KeyboardInput.W,
            carb.input.KeyboardInput.S,
            carb.input.KeyboardInput.A,
            carb.input.KeyboardInput.D,
            carb.input.KeyboardInput.R,
            carb.input.KeyboardInput.U,
            carb.input.KeyboardInput.J,
        }
        if event.input not in controlled_keys:
            return True

        if (
            event.input == carb.input.KeyboardInput.R
            and event.type == carb.input.KeyboardEventType.KEY_PRESS
        ):
            self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed_keys.discard(event.input)
        return True

    def _control_pressed(self):
        """读取左右Ctrl当前状态，用于切换高速档。"""

        left = self.input_interface.get_keyboard_value(
            self.keyboard,
            carb.input.KeyboardInput.LEFT_CONTROL,
        )
        right = self.input_interface.get_keyboard_value(
            self.keyboard,
            carb.input.KeyboardInput.RIGHT_CONTROL,
        )
        return bool(left or right)

    def update(self, step_dt):
        """按当前按键状态更新并返回一条三维指令。"""

        fast = self._control_pressed()
        forward_speed = FAST_FORWARD_SPEED if fast else FORWARD_SPEED
        yaw_speed = FAST_YAW_SPEED if fast else YAW_SPEED

        forward_direction = int(
            carb.input.KeyboardInput.W in self.pressed_keys
        ) - int(carb.input.KeyboardInput.S in self.pressed_keys)
        yaw_direction = int(
            carb.input.KeyboardInput.A in self.pressed_keys
        ) - int(carb.input.KeyboardInput.D in self.pressed_keys)

        self.command[0] = forward_direction * forward_speed
        self.command[1] = yaw_direction * yaw_speed

        leg_direction = int(
            carb.input.KeyboardInput.U in self.pressed_keys
        ) - int(carb.input.KeyboardInput.J in self.pressed_keys)
        self.command[2] = float(
            np.clip(
                self.command[2] + leg_direction * LEG_LENGTH_RATE * step_dt,
                self.leg_length_min,
                self.leg_length_max,
            )
        )
        return tuple(self.command)

    def consume_reset_request(self):
        """读取一次手动重置请求，并把控制指令恢复为初始值。"""

        if not self.reset_requested:
            return False
        self.reset_requested = False
        self.command = list(self.initial_command)
        self.pressed_keys.clear()
        return True

    def close(self):
        """取消键盘事件订阅。"""

        self.input_interface.unsubscribe_to_keyboard_events(
            self.keyboard,
            self.subscription,
        )


def main():
    env = None
    keyboard_controller = None

    try:
        # 根据任务注册信息自动加载环境配置。
        env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
        env_cfg.scene.num_envs = NUM_ENVS
        env_cfg.scene.clone_in_fabric = False
        env_cfg.seed = SEED

        # 播放时不按episode时长、跌倒、倾斜或接触条件自动重置。
        # 数值出现NaN/Inf时仍保留环境自身的安全重置。
        env_cfg.episode_length_s = 1.0e9
        if hasattr(env_cfg, "minimum_base_height"):
            env_cfg.minimum_base_height = -float("inf")
        if hasattr(env_cfg, "maximum_base_height"):
            env_cfg.maximum_base_height = float("inf")
        if hasattr(env_cfg, "termination_gravity_z"):
            env_cfg.termination_gravity_z = 2.0
        if hasattr(env_cfg, "illegal_contact_force_threshold"):
            env_cfg.illegal_contact_force_threshold = float("inf")

        env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=RENDER_MODE)
        base_env = env.unwrapped

        session, input_name, output_name = load_policy(
            MODEL_PATH,
            NUM_ENVS,
            env_cfg.observation_space,
            env_cfg.action_space,
        )

        if not HEADLESS:
            base_env.sim.set_camera_view(
                eye=CAMERA_EYE,
                target=CAMERA_TARGET,
            )

        observations, _ = env.reset()

        if KEYBOARD_CONTROL:
            if HEADLESS:
                raise RuntimeError("Keyboard control requires HEADLESS=False.")
            keyboard_controller = KeyboardCommandController(
                initial_command=INITIAL_COMMAND,
                leg_length_range=base_env.cfg.command_leg_length_range,
            )

        active_command = INITIAL_COMMAND
        apply_command(base_env, active_command)

        # reset()返回的观测可能仍包含随机指令；覆盖指令后重新生成观测。
        policy_observations = base_env._get_observations()["policy"]
        expected_observation_shape = (NUM_ENVS, env_cfg.observation_space)
        expected_action_shape = (NUM_ENVS, env_cfg.action_space)

        print(f"[INFO] Task: {TASK_NAME}")
        print(f"[INFO] Number of environments: {NUM_ENVS}")
        print(f"[INFO] Device: {base_env.device}")
        print(f"[INFO] Observation shape: {tuple(policy_observations.shape)}")
        print(f"[INFO] Policy frequency: {1.0 / base_env.step_dt:.1f} Hz")
        print(f"[INFO] Initial command: {INITIAL_COMMAND}")
        if keyboard_controller is not None:
            print("[CONTROL] W/S: +0.8/-0.8 m/s")
            print("[CONTROL] Ctrl+W/S: +1.5/-1.5 m/s")
            print("[CONTROL] A/D: counterclockwise/clockwise at 0.4 rad/s")
            print("[CONTROL] Ctrl+A/D: counterclockwise/clockwise at 0.8 rad/s")
            print("[CONTROL] Hold U/J: extend/retract legs")
            print("[CONTROL] R: reset robot")

        total_terminated = 0
        total_time_outs = 0

        step = 0
        while simulation_app.is_running():
            step += 1

            # 只在用户按R时主动重置机器人和控制指令。
            if (
                keyboard_controller is not None
                and keyboard_controller.consume_reset_request()
            ):
                observations, _ = env.reset()
                active_command = tuple(keyboard_controller.command)
                apply_command(base_env, active_command)
                policy_observations = base_env._get_observations()["policy"]
                print("[INFO] Manual reset: robot and command were reset.")

            # 根据键盘状态生成实时指令；未启用键盘时保持固定指令。
            if keyboard_controller is not None:
                active_command = keyboard_controller.update(base_env.step_dt)

            # 每步覆盖实时指令，并重新生成包含该指令的策略观测。
            if active_command is not None:
                apply_command(base_env, active_command)
                policy_observations = base_env._get_observations()["policy"]

            actions = run_policy(
                session,
                input_name,
                output_name,
                policy_observations,
                expected_observation_shape,
                expected_action_shape,
                base_env.device,
            )

            observations, rewards, terminated, time_outs, _ = env.step(actions)
            policy_observations = observations["policy"]

            terminated_count = int(torch.count_nonzero(terminated).item())
            time_out_count = int(torch.count_nonzero(time_outs).item())
            total_terminated += terminated_count
            total_time_outs += time_out_count

            if step == 1 or step % PRINT_INTERVAL == 0 or terminated_count > 0:
                base_height = (
                    base_env.robot.data.root_pos_w[:, 2]
                    - base_env.scene.env_origins[:, 2]
                )
                gravity = (
                    base_env.robot.data.projected_gravity_b[0]
                    .detach()
                    .cpu()
                    .tolist()
                )
                command_values = (
                    list(active_command)
                    if active_command is not None
                    else base_env.commands[0].detach().cpu().tolist()
                )
                action_values = actions[0].detach().cpu().tolist()

                print(
                    f"[STATE] step={step:4d} "
                    f"height_mean={base_height.mean().item():.4f} "
                    f"reward_mean={rewards.mean().item():.4f} "
                    f"gravity_0={[round(x, 3) for x in gravity]} "
                    f"command={[round(x, 3) for x in command_values]} "
                    f"action_0={[round(x, 3) for x in action_values]} "
                    f"terminated={terminated_count} "
                    f"time_out={time_out_count}"
                )

        print("[INFO] Playback finished.")
        print(f"[INFO] Terminated resets: {total_terminated}")
        print(f"[INFO] Time-out resets: {total_time_outs}")

    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
