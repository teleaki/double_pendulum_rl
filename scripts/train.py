from isaaclab.app import AppLauncher


# 训练时不打开GUI。
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


# Isaac Lab相关模块必须在AppLauncher之后导入。
from datetime import datetime
from pathlib import Path

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx
from isaaclab_tasks.utils import load_cfg_from_registry

# 导入根包，执行Gym任务注册。
import double_pendulum_rl  

# 训练参数
TASK_NAME = "WheelLeg-Balance-Direct-v0"
MODEL_NAME = "wheel_leg_balance.onnx"
NUM_ENVS = 256
NUM_TRAINING_ITERATIONS = 1000
RESUME_TRAINING = False
RESUME_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "rsl_rl"
    / "wheel_leg_balance_direct"
    / "2026-08-26_21-17-09"
    / "model_700.pt"
)

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

        # 第一轮只创建16个环境。
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
        print(f"[INFO] Training iterations: {NUM_TRAINING_ITERATIONS}")

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
            num_learning_iterations=(NUM_TRAINING_ITERATIONS),
            init_at_random_ep_len=True,
        )

        # runner.learn()结束时会自动保存最终的PT检查点。
        final_checkpoint_path = (
            log_dir / f"model_{runner.current_learning_iteration}.pt"
        )

        # 将最终策略切换到推理模式并导出为ONNX。
        runner.eval_mode()

        output_dir = project_root / "output"

        export_policy_as_onnx(
            policy=runner.alg.policy,
            normalizer=runner.alg.policy.actor_obs_normalizer,
            path=str(output_dir),
            filename=MODEL_NAME,
        )

        print(f"[PASS] Final PT model saved to: {final_checkpoint_path}")
        print(f"[PASS] ONNX policy exported to: {output_dir / MODEL_NAME}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
