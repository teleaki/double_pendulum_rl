from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 测试配置
# ------------------------------------------------------------

# 通过Gym任务名选择需要测试的环境：
# "WheelLeg-Balance-Direct-v0"：原地平衡环境
# "WheelLeg-Move-Direct-v0"：移动、旋转和腿高指令环境
TASK_NAME = "WheelLeg-Move-Direct-v0"

NUM_ENVS = 16
NUM_TEST_STEPS = 1200
SEED = 42

# "random"使用小幅随机动作；"zero"对全部关节施加零力矩。
ACTION_MODE = "random"
ACTION_NOISE_SCALE = 0.05

PRINT_INTERVAL = 60
HEADLESS = False
RENDER_MODE = None if HEADLESS else "human"


# Isaac Lab相关模块必须在AppLauncher之后导入。
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import carb
import omni.appwindow

# 导入根包以注册所有Gym任务。
import double_pendulum_rl  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


def main():
    """按照顶部测试配置创建并检查指定的强化学习环境。"""

    if ACTION_MODE not in {"random", "zero"}:
        raise ValueError(
            f"Unsupported ACTION_MODE={ACTION_MODE!r}; expected 'random' or 'zero'."
        )

    # 固定随机种子，方便比较多次测试结果。
    torch.manual_seed(SEED)

    # 根据TASK_NAME从Gym注册信息中加载对应的环境配置类。
    env_cfg = load_cfg_from_registry(
        TASK_NAME,
        "env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = SEED

    # 自定义URDF在GUI中使用Fabric克隆时可能显示不完整，因此测试时关闭。
    env_cfg.scene.clone_in_fabric = False

    env = gym.make(
        TASK_NAME,
        cfg=env_cfg,
        render_mode=RENDER_MODE,
    )
    base_env = env.unwrapped

    if not HEADLESS:
        # 默认视角可以完整显示4x4环境网格。
        base_env.sim.set_camera_view(
            eye=(7.0, 7.0, 5.0),
            target=(0.0, 0.0, 0.4),
        )

    # 键盘回调只记录重置请求，实际重置放在仿真循环中执行。
    reset_requested = [False]

    def on_keyboard_event(event, *args):
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == carb.input.KeyboardInput.R
        ):
            reset_requested[0] = True
        return True

    keyboard_subscription = None
    input_interface = None
    keyboard = None
    if not HEADLESS:
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

        print(f"[INFO] Environment created: {TASK_NAME}")
        print(f"[INFO] Number of environments: {base_env.num_envs}")
        print(f"[INFO] Action shape: ({base_env.num_envs}, {env_cfg.action_space})")
        print(f"[INFO] Observation shape: {tuple(policy_observations.shape)}")
        print(f"[INFO] Device: {base_env.device}")
        print(
            f"[INFO] Running {NUM_TEST_STEPS} policy steps with "
            f"action mode {ACTION_MODE!r}."
        )
        print("[INFO] Falling and automatic resets are expected in this test.")
        if not HEADLESS:
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

            if ACTION_MODE == "random":
                actions = ACTION_NOISE_SCALE * torch.randn(
                    (base_env.num_envs, env_cfg.action_space),
                    dtype=torch.float32,
                    device=base_env.device,
                )
                actions.clamp_(-1.0, 1.0)
            else:
                actions = torch.zeros(
                    (base_env.num_envs, env_cfg.action_space),
                    dtype=torch.float32,
                    device=base_env.device,
                )

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

                # MoveEnv提供commands；BalanceEnv没有该字段。
                if hasattr(base_env, "commands"):
                    command_mean = base_env.commands.mean(dim=0).tolist()
                    print(
                        "[COMMAND] mean="
                        f"{[round(value, 3) for value in command_mean]}"
                    )

        print("[PASS] Environment smoke test finished without NaN or Inf.")
        print(f"[INFO] Total terminated resets: {total_terminated}")
        print(f"[INFO] Total time-out resets: {total_time_outs}")

    finally:
        if keyboard_subscription is not None:
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
