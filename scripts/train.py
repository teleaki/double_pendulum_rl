from pathlib import Path

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 训练配置
# ------------------------------------------------------------

TASK_NAME = "WheelLeg-Move-Direct-v0"
MODEL_BASENAME = "wheel_leg_move_v2"
NUM_ENVS = 256
NUM_TRAINING_ITERATIONS = None
RESUME_TRAINING = True
RESUME_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "pre_models" / "model_2999.pt"
)
HEADLESS = False


def get_available_model_paths(output_dir: Path, model_basename: str):
    """返回同一版本且不会覆盖已有文件的PT/ONNX路径。

    只要同版本的PT或ONNX任意一个已存在，就为两者一起使用
    下一个序号，避免训练检查点和推理模型版本错位。
    """

    index = 0
    while True:
        suffix = "" if index == 0 else f" ({index})"
        pt_path = output_dir / f"{model_basename}{suffix}.pt"
        onnx_path = output_dir / f"{model_basename}{suffix}.onnx"
        if not pt_path.exists() and not onnx_path.exists():
            return pt_path, onnx_path
        index += 1


# 训练时不打开GUI，减少渲染开销并提高并行仿真速度。
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app


# Isaac Lab相关模块必须在AppLauncher之后导入。
from datetime import datetime

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import load_cfg_from_registry

# 导入根包，执行Gym任务注册。
import double_pendulum_rl


def main():
    env = None

    try:
        # 通过Gym注册表读取环境配置。
        env_cfg = load_cfg_from_registry(
            TASK_NAME,
            "env_cfg_entry_point",
        )

        # 通过Gym注册表读取RSL-RL PPO配置。
        agent_cfg = load_cfg_from_registry(
            TASK_NAME,
            "rsl_rl_cfg_entry_point",
        )

        # 使用训练脚本顶部配置的并行环境数量。
        env_cfg.scene.num_envs = NUM_ENVS

        # 环境、物理仿真和PPO网络使用相同设备。
        env_cfg.sim.device = agent_cfg.device

        # 环境和PPO使用相同随机种子。
        env_cfg.seed = agent_cfg.seed

        # 提前检查配置是否有缺失字段。
        env_cfg.validate()
        agent_cfg.validate()

        # 创建Gym环境。
        env = gym.make(
            TASK_NAME,
            cfg=env_cfg,
            render_mode=None,
        )

        # 将Isaac Lab环境转换成RSL-RL需要的接口。
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        # 创建本次训练的日志目录。
        project_root = Path(__file__).resolve().parents[1]

        log_root = project_root / "logs" / "rsl_rl" / agent_cfg.experiment_name

        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Task: {TASK_NAME}")
        print(f"[INFO] Number of environments: {env.num_envs}")
        print(f"[INFO] Training device: {agent_cfg.device}")
        print(f"[INFO] Log directory: {log_dir}")
        training_iterations = (
            agent_cfg.max_iterations
            if NUM_TRAINING_ITERATIONS is None
            else NUM_TRAINING_ITERATIONS
        )

        print(f"[INFO] Training iterations: {training_iterations}")

        # 保存本次训练使用的配置。
        dump_yaml(
            str(log_dir / "params" / "env.yaml"),
            env_cfg,
        )

        dump_yaml(
            str(log_dir / "params" / "agent.yaml"),
            agent_cfg,
        )

        # 创建PPO训练器。
        runner = OnPolicyRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=str(log_dir),
            device=agent_cfg.device,
        )

        # 根据训练参数决定是否从已有的PT检查点继续训练。
        if RESUME_TRAINING:
            if RESUME_CHECKPOINT_PATH is None:
                raise ValueError(
                    "Set RESUME_CHECKPOINT_PATH when RESUME_TRAINING=True."
                )
            if not RESUME_CHECKPOINT_PATH.is_file():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {RESUME_CHECKPOINT_PATH}"
                )

            runner.load(
                str(RESUME_CHECKPOINT_PATH),
                load_optimizer=True,
            )

            print(f"[INFO] Resuming training from: {RESUME_CHECKPOINT_PATH}")

        # 开始训练。
        runner.learn(
            num_learning_iterations=training_iterations,
            init_at_random_ep_len=True,
        )

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # PT和ONNX使用相同版本名，且不覆盖已有输出。
        output_pt_path, output_onnx_path = get_available_model_paths(
            output_dir,
            MODEL_BASENAME,
        )

        # PT保留Actor、Critic、优化器和训练状态，可用于继续训练。
        runner.save(str(output_pt_path))

        # ONNX只用于策略推理和部署。
        runner.eval_mode()

        export_policy_as_onnx(
            policy=runner.alg.policy,
            normalizer=runner.alg.policy.actor_obs_normalizer,
            path=str(output_dir),
            filename=output_onnx_path.name,
        )

        print(f"[PASS] Trainable PT checkpoint saved to: {output_pt_path}")
        print(f"[PASS] ONNX policy exported to: {output_onnx_path}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
