from pathlib import Path

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 播放配置
# ------------------------------------------------------------

# 任务和模型必须匹配：
# Balance：WheelLeg-Balance-Direct-v0 + wheel_leg_balance.onnx
# Move：   WheelLeg-Move-Direct-v0    + wheel_leg_move.onnx
TASK_NAME = "WheelLeg-Move-Direct-v0"
MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "output" / "wheel_leg_move.onnx"
)
NUM_ENVS = 1
NUM_STEPS = 3600
PRINT_INTERVAL = 60
SEED = 42

HEADLESS = False
RENDER_MODE = None if HEADLESS else "human"
CAMERA_EYE = (3.0, 3.0, 1.8)
CAMERA_TARGET = (0.0, 0.0, 0.5)

# Move环境固定指令：[前向速度m/s, 偏航角速度rad/s, 腿长m]。
# None表示使用训练时的随机指令；播放Balance模型时必须设为None。
FIXED_COMMAND = (0.3, 0.0, 0.30)


# Isaac Lab相关模块必须在AppLauncher之后导入。
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

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


def apply_fixed_command(base_env):
    """给命令式环境的全部并行机器人写入同一条固定指令。"""

    if FIXED_COMMAND is None:
        return
    if not hasattr(base_env, "commands"):
        raise RuntimeError(
            "FIXED_COMMAND is only supported by command-based environments. "
            "Set FIXED_COMMAND=None for Balance."
        )
    if len(FIXED_COMMAND) != base_env.commands.shape[1]:
        raise RuntimeError(
            f"Expected {base_env.commands.shape[1]} command values, "
            f"got {len(FIXED_COMMAND)}"
        )

    command = torch.tensor(
        FIXED_COMMAND,
        dtype=torch.float32,
        device=base_env.device,
    )
    base_env.commands[:] = command

    # 禁止MoveEnv在播放过程中按训练周期重新随机采样指令。
    if hasattr(base_env, "command_time_left"):
        base_env.command_time_left.fill_(float("inf"))


def main():
    env = None

    try:
        # 根据任务注册信息自动加载BalanceEnvCfg或MoveEnvCfg。
        env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
        env_cfg.scene.num_envs = NUM_ENVS
        env_cfg.scene.clone_in_fabric = False
        env_cfg.seed = SEED

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
        apply_fixed_command(base_env)

        # reset()返回的观测可能仍包含随机指令；覆盖指令后重新生成观测。
        policy_observations = base_env._get_observations()["policy"]
        expected_observation_shape = (NUM_ENVS, env_cfg.observation_space)
        expected_action_shape = (NUM_ENVS, env_cfg.action_space)

        print(f"[INFO] Task: {TASK_NAME}")
        print(f"[INFO] Number of environments: {NUM_ENVS}")
        print(f"[INFO] Device: {base_env.device}")
        print(f"[INFO] Observation shape: {tuple(policy_observations.shape)}")
        print(f"[INFO] Policy frequency: {1.0 / base_env.step_dt:.1f} Hz")
        print(f"[INFO] Fixed command: {FIXED_COMMAND}")

        total_terminated = 0
        total_time_outs = 0

        for step in range(1, NUM_STEPS + 1):
            if not simulation_app.is_running():
                break

            # 自动重置或定时采样后重新覆盖固定指令和对应观测。
            if FIXED_COMMAND is not None:
                apply_fixed_command(base_env)
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
                action_values = actions[0].detach().cpu().tolist()

                print(
                    f"[STATE] step={step:4d} "
                    f"height_mean={base_height.mean().item():.4f} "
                    f"reward_mean={rewards.mean().item():.4f} "
                    f"gravity_0={[round(x, 3) for x in gravity]} "
                    f"action_0={[round(x, 3) for x in action_values]} "
                    f"terminated={terminated_count} "
                    f"time_out={time_out_count}"
                )

        print("[INFO] Playback finished.")
        print(f"[INFO] Terminated resets: {total_terminated}")
        print(f"[INFO] Time-out resets: {total_time_outs}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
