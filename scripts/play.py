from isaaclab.app import AppLauncher


# 启动isaac sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

# 导入后执行 Gym 环境注册。
import double_pendulum_rl.env 
from double_pendulum_rl.env.balance_env import BalanceEnvCfg

TASK_NAME = "WheelLeg-Balance-Direct-v0"
MODEL_PATH = Path(__file__).resolve().parents[1] / "output" / "wheel_leg_balance.onnx"
NUM_STEPS = 3600
PRINT_INTERVAL = 60

def load_policy(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Input: name={input_info.name}, shape={input_info.shape}")
    print(f"[INFO] Output: name={output_info.name}, shape={output_info.shape}")

    if input_info.shape != [1, 25]:
        raise RuntimeError(f"Expected ONNX input shape [1, 25], got {input_info.shape}")

    if output_info.shape != [1, 6]:
        raise RuntimeError(
            f"Expected ONNX output shape [1, 6], got {output_info.shape}"
        )

    return session, input_info.name, output_info.name

def run_policy(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    observations: torch.Tensor,
    device: str,
) -> torch.Tensor:
    if observations.shape != (1, 25):
        raise RuntimeError(
            f"Expected observation shape (1, 25), got {tuple(observations.shape)}"
        )

    if not torch.isfinite(observations).all():
        raise RuntimeError("Observation contains NaN or Inf.")

    observations_numpy = (
        observations
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    actions_numpy = session.run(
        [output_name],
        {input_name: observations_numpy},
    )[0]

    actions = torch.from_numpy(actions_numpy).to(
        device=device,
        dtype=torch.float32,
    )

    if not torch.isfinite(actions).all():
        raise RuntimeError("Policy output contains NaN or Inf.")

    return torch.clamp(actions, -1.0, 1.0)

def main():
    env = None

    try:
        session, input_name, output_name = load_policy(MODEL_PATH)

        env_cfg = BalanceEnvCfg()

        # ONNX 输入固定为 [1, 25]，因此必须只创建一个环境。
        env_cfg.scene.num_envs = 1
        env_cfg.scene.clone_in_fabric = False
        env_cfg.seed = 42

        env = gym.make(
            TASK_NAME,
            cfg=env_cfg,
            render_mode="human",
        )

        base_env = env.unwrapped

        base_env.sim.set_camera_view(
            eye=(3.0, 3.0, 1.8),
            target=(0.0, 0.0, 0.5),
        )

        observations, _ = env.reset()
        policy_observations = observations["policy"]

        print("[INFO] Isaac Sim policy playback started.")
        print(f"[INFO] Device: {base_env.device}")
        print(f"[INFO] Observation shape: {tuple(policy_observations.shape)}")
        print(f"[INFO] Policy frequency: {1.0 / base_env.step_dt:.1f} Hz")

        total_terminated = 0
        total_time_outs = 0

        for step in range(1, NUM_STEPS + 1):
            if not simulation_app.is_running():
                break

            actions = run_policy(
                session=session,
                input_name=input_name,
                output_name=output_name,
                observations=policy_observations,
                device=base_env.device,
            )

            (
                observations,
                rewards,
                terminated,
                time_outs,
                _,
            ) = env.step(actions)

            policy_observations = observations["policy"]

            terminated_count = int(
                torch.count_nonzero(terminated).item()
            )
            time_out_count = int(
                torch.count_nonzero(time_outs).item()
            )

            total_terminated += terminated_count
            total_time_outs += time_out_count

            if (
                step == 1
                or step % PRINT_INTERVAL == 0
                or terminated_count > 0
            ):
                base_height = (
                    base_env.robot.data.root_pos_w[0, 2]
                    - base_env.scene.env_origins[0, 2]
                )

                gravity = (
                    base_env.robot.data.projected_gravity_b[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                action_values = (
                    actions[0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                print(
                    f"[STATE] step={step:4d} "
                    f"height={base_height.item():.4f} "
                    f"reward={rewards[0].item():.4f} "
                    f"gravity={[round(x, 3) for x in gravity]} "
                    f"action={[round(x, 3) for x in action_values]} "
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
