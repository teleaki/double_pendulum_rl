from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每台机器人在进行一次PPO更新前采集多少个策略步
    num_steps_per_env = 24
    # PPO最大更新次数
    max_iterations = 1000
    # 每50次更新保存一次模型
    save_interval = 50
    # 实验名称
    experiment_name = "wheel_leg_balance_direct"
    # 观测映射
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    # 策略网络与价值网络。
    policy = RslRlPpoActorCriticCfg(
        # 初始动作探索噪声。
        init_noise_std=0.3,
        # 观测已经在BalanceEnv中手动缩放，暂时不再归一化。
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Actor：观测 → 动作。
        actor_hidden_dims=[128, 128, 64],
        # Critic：观测 → 状态价值。
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    # PPO算法参数。
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失权重。
        value_loss_coef=1.0,
        # 对价值函数更新也使用裁剪。
        use_clipped_value_loss=True,
        # PPO概率比裁剪范围。
        clip_param=0.2,
        # 鼓励探索的熵奖励权重。
        entropy_coef=0.005,
        # 同一批数据重复学习5轮。
        num_learning_epochs=5,
        # 每轮把采样数据分成4个mini-batch。
        num_mini_batches=4,
        # 初始学习率。
        learning_rate=3.0e-4,
        # 根据KL散度自动调整学习率。
        schedule="adaptive",
        # 奖励折扣系数。
        gamma=0.99,
        # GAE优势估计参数。
        lam=0.95,
        # 自适应学习率的目标KL散度。
        desired_kl=0.01,
        # 梯度裁剪，防止更新爆炸。
        max_grad_norm=1.0,
    )


@configclass
class MovePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每台机器人在进行一次PPO更新前采集32个策略步。
    # 策略频率60Hz时，一段rollout约为0.53秒。
    num_steps_per_env = 32

    # PPO最大更新次数。
    max_iterations = 3000

    # 每50次更新保存一次模型。
    save_interval = 50

    # Move任务使用独立的实验名和日志目录。
    experiment_name = "wheel_leg_move_direct"

    # Actor和Critic当前使用相同的28维策略观测。
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # 策略网络与价值网络配置。
    policy = RslRlPpoActorCriticCfg(
        # 纯力矩控制对过大的初始噪声较敏感，先使用保守值。
        init_noise_std=0.3,

        # MoveEnv已经手动缩放观测，第一版不再重复归一化。
        actor_obs_normalization=False,
        critic_obs_normalization=False,

        # Move任务同时包含平衡、移动、转向和高度控制，
        # 网络容量比Balance任务略大。
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # PPO算法参数。
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失权重。
        value_loss_coef=1.0,

        # 对价值函数更新使用裁剪。
        use_clipped_value_loss=True,

        # PPO概率比裁剪范围。
        clip_param=0.2,

        # 保留适量探索，避免策略过早收敛到只会站立。
        entropy_coef=0.005,

        # 每批采样数据重复训练5轮。
        num_learning_epochs=5,

        # 256环境、每环境32步时总批量8192，
        # 分成4个mini-batch后每批2048个样本。
        num_mini_batches=4,

        # 初始学习率，由adaptive策略根据KL自动调整。
        learning_rate=3.0e-4,
        schedule="adaptive",

        # 60Hz纯力矩控制需要关注较长期的平衡结果，
        # 因此使用比Balance任务更大的折扣系数。
        gamma=0.995,
        lam=0.95,

        # 自适应学习率的目标KL散度。
        desired_kl=0.01,

        # 梯度裁剪，防止更新爆炸。
        max_grad_norm=1.0,
    )
