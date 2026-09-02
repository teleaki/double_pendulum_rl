import math
import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
)

from double_pendulum_rl.robot import ROBOT_CFG


@configclass
class BalanceEnvCfg(DirectRLEnvCfg):
    # =========================================================
    # Fixed interface configuration
    # =========================================================

    decimation = 2  # 策略频率 120 / 2 = 60Hz

    # 6维策略输出：
    # [左髋, 右髋, 左膝, 右膝, 左轮, 右轮]
    action_space = 6

    # 28维观测空间：
    # 重力投影 3, 车身角速度 3, 车身线速度 3,
    # 腿关节位置 4, 腿关节速度 4, 轮子速度 2,
    # 运动指令 3, 上一次输出 6
    observation_space = 28

    # 物理仿真
    sim = SimulationCfg(
        dt=1.0 / 120.0,  # 物理频率120Hz
        render_interval=decimation,
    )

    # 并行场景
    scene = InteractiveSceneCfg(
        num_envs=16,  # 并行机器人数量
        env_spacing=2.0,  # 相邻环境间距
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # 地面配置
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,  # 静摩擦 1.0
            dynamic_friction=0.8,  # 动摩擦 0.8
            restitution=0.0,  # 恢复系数 0
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )

    # 传感器配置
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )

    illegal_contact_force_threshold = 5.0

    # 机器人配置
    robot_cfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 动作参数
    action_clip = 1.0  # 归一化输出
    leg_effort_scale = 40.0  # 腿电机最大力矩
    wheel_effort_scale = 5.0  # 轮电机最大力矩

    # 观测参数
    gravity_obs_scale = 1.0
    angular_velocity_obs_scale = 0.25
    linear_velocity_obs_scale = 1.0
    leg_position_obs_scale = 1.0
    leg_velocity_obs_scale = 0.1
    wheel_velocity_obs_scale = 0.05

    command_forward_obs_scale = 1.0
    command_yaw_obs_scale = 1.0
    command_leg_length_obs_scale = 10.0
    command_leg_length_center = 0.35

    # =========================================================
    # Task configuration
    # =========================================================

    episode_length_s = 10.0  # episode最长时间10s

    # Balance任务不变化指令，但保留项目统一的三维指令接口。
    command_forward_velocity_range = (0.0, 0.0)
    command_yaw_velocity_range = (0.0, 0.0)
    command_leg_length_range = (0.35, 0.35)
    command_resampling_time_s = 3.0

    # 重置随机化
    initial_pitch_range = math.radians(2.0)
    initial_leg_position_noise = 0.02
    initial_joint_velocity_noise = 0.05

    # 终止参数
    minimum_base_height = 0.45  # 车身太低
    maximum_base_height = 0.75  # 车身太高
    termination_gravity_z = -0.5  # 车身倾斜过大

    # 目标状态
    target_base_height = 0.584

    # 奖励权重
    reward_alive = 1.0
    reward_upright = 2.0
    reward_base_height = 1.0
    base_height_error_scale = 20.0

    penalty_linear_velocity = -0.5
    penalty_yaw_velocity = -0.1
    penalty_leg_deviation = -0.2
    penalty_joint_velocity = -0.01
    penalty_action = -0.01
    penalty_action_rate = -0.05
    penalty_joint_limit = -1.0
    penalty_termination = -5.0


class BalanceEnv(DirectRLEnv):
    cfg: BalanceEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # 创建仿真、场景和机器人
        super().__init__(cfg, render_mode, **kwargs)

        # 按照明确顺序查找关节，保证动作含义固定。
        self.hip_joint_ids, self.hip_joint_names = self.robot.find_joints(
            [
                "left_hip_joint",
                "right_hip_joint",
            ],
            preserve_order=True,
        )
        self.knee_joint_ids, self.knee_joint_names = self.robot.find_joints(
            [
                "left_knee_joint",
                "right_knee_joint",
            ],
            preserve_order=True,
        )
        self.wheel_joint_ids, self.wheel_joint_names = self.robot.find_joints(
            [
                "left_wheel_joint",
                "right_wheel_joint",
            ],
            preserve_order=True,
        )
        if len(self.hip_joint_ids) != 2:
            raise RuntimeError(f"髋关节数量错误：{self.hip_joint_names}")
        if len(self.knee_joint_ids) != 2:
            raise RuntimeError(f"膝关节数量错误：{self.knee_joint_names}")
        if len(self.wheel_joint_ids) != 2:
            raise RuntimeError(f"轮子关节数量错误：{self.wheel_joint_names}")

        # [左髋, 右髋, 左膝, 右膝]
        self.leg_joint_ids = self.hip_joint_ids + self.knee_joint_ids

        (
            self.illegal_contact_body_ids,
            self.illegal_contact_body_names,
        ) = self.contact_sensor.find_bodies(
            [
                "base_link",
                ".*upper_leg",
                ".*lower_leg",
            ]
        )

        # 腿部目标姿态
        self.default_leg_pos = self.robot.data.default_joint_pos[
            :, self.leg_joint_ids
        ].clone()

        # 保存上一帧动作
        self.previous_actions = torch.zeros_like(self.actions)

        # 六个关节最终施加的实际力矩
        self.joint_efforts = torch.zeros(
            (self.num_envs, self.robot.num_joints),
            dtype=torch.float32,
            device=self.device,
        )

        self.commands = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self.command_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.command_resample_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _setup_scene(self):
        # 创建机器人
        self.robot = Articulation(self.cfg.robot_cfg)
        # 注册到场景
        self.scene.articulations["robot"] = self.robot
        # 配置传感器
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)

        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # 创建地面
        self.cfg.ground_cfg.func(
            "/World/Ground",
            self.cfg.ground_cfg,
        )
        # 克隆并行环境
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])

        # 添加环境光
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _sample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return

        sampled_commands = torch.empty(
            (env_ids.numel(), 3), dtype=torch.float32, device=self.device
        )
        sampled_commands[:, 0].uniform_(
            *self.cfg.command_forward_velocity_range
        )
        sampled_commands[:, 1].uniform_(
            *self.cfg.command_yaw_velocity_range
        )
        sampled_commands[:, 2].uniform_(
            *self.cfg.command_leg_length_range
        )
        self.commands[env_ids] = sampled_commands
        self.command_time_left[env_ids] = self.cfg.command_resampling_time_s

    def _pre_physics_step(self, actions):
        # 更新上帧输出
        self.previous_actions.copy_(self.actions)

        # 处理NaN
        safe_actions = torch.nan_to_num(
            actions,
            nan=0.0,
            posinf=self.cfg.action_clip,
            neginf=-self.cfg.action_clip,
        )

        # 将策略动作限制在 [-1, 1]。
        self.actions.copy_(
            torch.clamp(
                safe_actions,
                min=-self.cfg.action_clip,
                max=self.cfg.action_clip,
            )
        )

        self.command_time_left -= self.step_dt
        self.command_resample_pending |= self.command_time_left <= 0.0

    def _apply_action(self):
        # 清空旧力矩
        self.joint_efforts.zero_()
        # 控制6个关节
        self.joint_efforts[:, self.hip_joint_ids] = (
            self.actions[:, 0:2] * self.cfg.leg_effort_scale
        )
        self.joint_efforts[:, self.knee_joint_ids] = (
            self.actions[:, 2:4] * self.cfg.leg_effort_scale
        )
        self.joint_efforts[:, self.wheel_joint_ids] = (
            self.actions[:, 4:6] * self.cfg.wheel_effort_scale
        )

        # 写入目标缓存
        self.robot.set_joint_effort_target(self.joint_efforts)

    def _get_observations(self):
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        leg_pos_error = joint_pos[:, self.leg_joint_ids] - self.default_leg_pos

        state_obs = torch.cat(
            (
                # 3维：重力在车身坐标系中的投影。
                self.robot.data.projected_gravity_b * self.cfg.gravity_obs_scale,
                # 3维：车身角速度。
                self.robot.data.root_ang_vel_b * self.cfg.angular_velocity_obs_scale,
                # 3维：车身线速度。
                self.robot.data.root_lin_vel_b * self.cfg.linear_velocity_obs_scale,
                # 4维：腿关节相对默认位置的偏差。
                leg_pos_error * self.cfg.leg_position_obs_scale,
                # 4维：腿关节速度。
                joint_vel[:, self.leg_joint_ids] * self.cfg.leg_velocity_obs_scale,
                # 2维：轮子速度。
                joint_vel[:, self.wheel_joint_ids] * self.cfg.wheel_velocity_obs_scale,
            ),
            dim=-1,
        )

        command_obs = torch.stack(
            (
                self.commands[:, 0] * self.cfg.command_forward_obs_scale,
                self.commands[:, 1] * self.cfg.command_yaw_obs_scale,
                (
                    self.commands[:, 2]
                    - self.cfg.command_leg_length_center
                )
                * self.cfg.command_leg_length_obs_scale,
            ),
            dim=-1,
        )

        observations = torch.cat(
            (state_obs, command_obs, self.actions),
            dim=-1,
        )

        return {
            "policy": observations,
        }

    def _get_dones(self):
        # 当前车身高度相对于所属并行环境原点。
        base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

        # 高度过低，通常表示车已经倒下。
        too_low = base_height < self.cfg.minimum_base_height

        # 高度异常升高。
        too_high = base_height > self.cfg.maximum_base_height

        # 正常站立时 projected_gravity_b 的 z 大约为 -1。
        # 数值变大代表车身倾斜。
        too_tilted = (
            self.robot.data.projected_gravity_b[:, 2] > self.cfg.termination_gravity_z
        )

        # 除轮以外的地方接地
        contact_forces = self.contact_sensor.data.net_forces_w_history

        illegal_contact = torch.any(
            torch.max(
                torch.norm(
                    contact_forces[
                        :,
                        :,
                        self.illegal_contact_body_ids,
                    ],
                    dim=-1,
                ),
                dim=1,
            ).values
            > self.cfg.illegal_contact_force_threshold,
            dim=1,
        )

        # 检查仿真是否出现 NaN 或无穷大。
        invalid_state = ~torch.isfinite(self.robot.data.root_state_w).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=1)

        terminated = too_low | too_high | too_tilted | invalid_state | illegal_contact

        # episode_length_buf 按策略步计数。
        # 10秒、60Hz时大约是600步。
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _get_rewards(self):
        projected_gravity_z = self.robot.data.projected_gravity_b[:, 2]
        base_height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_ids]
        leg_vel = self.robot.data.joint_vel[:, self.leg_joint_ids]

        # 还没有倒下时获得存活奖励。
        alive = (~self.reset_terminated).float()

        # 完全直立时 projected_gravity_z 约为 -1。
        upright = torch.clamp(
            -projected_gravity_z,
            min=0.0,
            max=1.0,
        ).square()

        # 高度越接近目标值，越接近1。
        height_error = base_height - self.cfg.target_base_height
        height_reward = torch.exp(
            -self.cfg.base_height_error_scale * height_error.square()
        )

        # 惩罚水平移动，暂时不惩罚竖直速度。
        linear_velocity_error = torch.sum(
            self.robot.data.root_lin_vel_b[:, 0:2].square(),
            dim=1,
        )

        # 惩罚绕竖直轴旋转。
        yaw_velocity_error = self.robot.data.root_ang_vel_b[:, 2].square()

        # 惩罚腿偏离默认姿态。
        leg_deviation_error = torch.sum(
            (leg_pos - self.default_leg_pos).square(),
            dim=1,
        )

        # 这里只惩罚腿关节速度。
        # 不惩罚轮速，否则轮子可能不愿意参与平衡。
        leg_velocity_error = torch.sum(
            leg_vel.square(),
            dim=1,
        )

        # 惩罚动作过大。
        action_error = torch.sum(
            self.actions.square(),
            dim=1,
        )

        # 惩罚动作变化过快。
        action_rate_error = torch.sum(
            (self.actions - self.previous_actions).square(),
            dim=1,
        )

        # 关节超过软限位时进行惩罚。
        leg_limits = self.robot.data.soft_joint_pos_limits[:, self.leg_joint_ids]

        lower_limit_error = torch.clamp(
            leg_limits[:, :, 0] - leg_pos,
            min=0.0,
        )

        upper_limit_error = torch.clamp(
            leg_pos - leg_limits[:, :, 1],
            min=0.0,
        )

        joint_limit_error = torch.sum(
            lower_limit_error.square() + upper_limit_error.square(),
            dim=1,
        )

        # 普通奖励按时间缩放。
        # 这样改变策略频率后，总奖励量级不会变化太大。
        rewards = {
            "alive": (alive * self.cfg.reward_alive * self.step_dt),
            "upright": (upright * self.cfg.reward_upright * self.step_dt),
            "base_height": (height_reward * self.cfg.reward_base_height * self.step_dt),
            "linear_velocity": (
                linear_velocity_error * self.cfg.penalty_linear_velocity * self.step_dt
            ),
            "yaw_velocity": (
                yaw_velocity_error * self.cfg.penalty_yaw_velocity * self.step_dt
            ),
            "leg_deviation": (
                leg_deviation_error * self.cfg.penalty_leg_deviation * self.step_dt
            ),
            "joint_velocity": (
                leg_velocity_error * self.cfg.penalty_joint_velocity * self.step_dt
            ),
            "action": (action_error * self.cfg.penalty_action * self.step_dt),
            "action_rate": (
                action_rate_error * self.cfg.penalty_action_rate * self.step_dt
            ),
            "joint_limit": (
                joint_limit_error * self.cfg.penalty_joint_limit * self.step_dt
            ),
            # 倒下是瞬时事件，不乘 step_dt。
            "termination": (
                self.reset_terminated.float() * self.cfg.penalty_termination
            ),
        }

        total_reward = torch.sum(
            torch.stack(list(rewards.values())),
            dim=0,
        )

        # 本步奖励已使用旧指令结算，再为下一步切换指令。
        resample_mask = self.command_resample_pending & ~self.reset_terminated
        resample_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
        self.command_resample_pending.zero_()
        self._sample_commands(resample_ids)

        # 防止异常仿真状态把 NaN 传入训练器。
        return torch.nan_to_num(
            total_reward,
            nan=self.cfg.penalty_termination,
            posinf=self.cfg.penalty_termination,
            neginf=self.cfg.penalty_termination,
        )

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )

        # 重置场景缓存、episode计数器等。
        super()._reset_idx(env_ids)

        # 清空动作历史。
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.joint_efforts[env_ids] = 0.0
        self.command_resample_pending[env_ids] = False

        # 清空关节力矩目标，避免重置前的力矩残留。
        self.robot.set_joint_effort_target(
            self.joint_efforts[env_ids],
            env_ids=env_ids,
        )

        # 复制默认根状态：
        # [位置3, 四元数4, 线速度3, 角速度3]
        root_state = self.robot.data.default_root_state[env_ids].clone()

        # 加上各个并行环境在世界坐标系中的原点。
        root_state[:, 0:3] += self.scene.env_origins[env_ids]

        num_reset_envs = env_ids.numel()

        # 随机生成初始俯仰角。
        initial_pitch = torch.empty(
            num_reset_envs,
            dtype=torch.float32,
            device=self.device,
        ).uniform_(
            -self.cfg.initial_pitch_range,
            self.cfg.initial_pitch_range,
        )

        zero_angle = torch.zeros_like(initial_pitch)

        pitch_quaternion = quat_from_euler_xyz(
            zero_angle,
            initial_pitch,
            zero_angle,
        )

        # Isaac Lab 四元数顺序为 [w, x, y, z]。
        root_state[:, 3:7] = quat_mul(
            root_state[:, 3:7],
            pitch_quaternion,
        )

        # 复制默认关节状态。
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()

        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 给四个腿关节增加很小的位置随机扰动。
        joint_pos[:, self.leg_joint_ids] += torch.empty(
            (
                num_reset_envs,
                len(self.leg_joint_ids),
            ),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(
            -self.cfg.initial_leg_position_noise,
            self.cfg.initial_leg_position_noise,
        )

        # 给所有六个关节增加很小的速度随机扰动。
        joint_vel += torch.empty_like(joint_vel).uniform_(
            -self.cfg.initial_joint_velocity_noise,
            self.cfg.initial_joint_velocity_noise,
        )

        # 防止随机化后的腿关节位置超过限位。
        leg_limits = self.robot.data.soft_joint_pos_limits[env_ids][
            :, self.leg_joint_ids
        ]

        leg_pos = joint_pos[:, self.leg_joint_ids]

        joint_pos[:, self.leg_joint_ids] = torch.maximum(
            torch.minimum(
                leg_pos,
                leg_limits[:, :, 1],
            ),
            leg_limits[:, :, 0],
        )

        # 将根状态写回仿真。
        self.robot.write_root_pose_to_sim(
            root_state[:, 0:7],
            env_ids=env_ids,
        )

        self.robot.write_root_velocity_to_sim(
            root_state[:, 7:13],
            env_ids=env_ids,
        )

        # 将关节状态写回仿真。
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

        self._sample_commands(env_ids)
