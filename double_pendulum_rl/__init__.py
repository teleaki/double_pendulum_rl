import gymnasium as gym


gym.register(
    id="WheelLeg-Balance-Direct-v0",
    entry_point="double_pendulum_rl.env.balance_env:BalanceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "double_pendulum_rl.env.balance_env:BalanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "double_pendulum_rl.agents.ppo_cfg:BalancePPORunnerCfg"
        ),
    },
)
