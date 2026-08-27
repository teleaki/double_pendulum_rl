import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from double_pendulum_rl.robot import ROBOT_CFG


@configclass
class MoveEnvCfg(DirectRLEnvCfg):
    """移动、转向、腿高指令跟踪环境配置。"""

    # 时间参数
    decimation = 2  # 策略频率 120 / 2 = 60Hz
    episode_length_s = 20.0  # episode最长时间20s

    # 6维策略输出：
    # [左髋, 右髋, 左膝, 右膝, 左轮, 右轮]
    # 六个关节全部使用纯力矩控制。
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
        num_envs=16,  # 默认并行机器人数量
        env_spacing=2.0,  # 相邻环境间距
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # 地面配置
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,  # 静摩擦
            dynamic_friction=0.8,  # 动摩擦
            restitution=0.0,  # 恢复系数，地面不弹跳
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )

    # 接触传感器用于检测车身和腿部是否非法接地。
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
    action_clip = 1.0  # 策略输出限制在[-1, 1]
    leg_effort_scale = 40.0  # 髋、膝最大控制力矩
    wheel_effort_scale = 5.0  # 轮子最大控制力矩

    # 观测缩放参数
    gravity_obs_scale = 1.0
    angular_velocity_obs_scale = 0.25
    linear_velocity_obs_scale = 1.0
    leg_position_obs_scale = 1.0
    leg_velocity_obs_scale = 0.1
    wheel_velocity_obs_scale = 0.05

    # 重置随机化参数
    initial_pitch_range = math.radians(2.0)
    initial_leg_position_noise = 0.02
    initial_joint_velocity_noise = 0.05

    # 指令采样参数：
    # [机体坐标系X方向速度, 绕机体Z轴角速度, 车身离地高度]
    command_forward_velocity_range = (-0.5, 0.5)
    command_yaw_velocity_range = (-0.8, 0.8)
    command_base_height_range = (0.50, 0.66)
    command_resampling_time_s = 3.0
    standing_command_probability = 0.15

    # 指令进入策略观测前的缩放参数
    command_forward_obs_scale = 2.0
    command_yaw_obs_scale = 1.0
    command_height_obs_scale = 5.0

    # 终止参数。合法高度范围需要覆盖全部高度指令。
    minimum_base_height = 0.38
    maximum_base_height = 0.78
    termination_gravity_z = -0.5

    # 默认高度只用作高度指令观测的中心值，不是固定控制目标。
    target_base_height = 0.584

    # 指令跟踪奖励权重和误差敏感度
    reward_forward_tracking = 2.0
    reward_yaw_tracking = 1.0
    reward_height_tracking = 1.5
    forward_tracking_error_scale = 4.0
    yaw_tracking_error_scale = 2.0
    height_tracking_error_scale = 30.0

    # 平衡奖励与动作、安全惩罚
    reward_alive = 0.5
    reward_upright = 2.0
    penalty_vertical_velocity = -0.2
    penalty_roll_pitch_velocity = -0.05
    penalty_leg_velocity = -0.005
    penalty_action = -0.002
    penalty_action_rate = -0.02
    penalty_joint_limit = -1.0
    penalty_termination = -5.0


class MoveEnv(DirectRLEnv):
    """在保持平衡的同时跟踪移动、转向和车身高度指令。"""

    cfg: MoveEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # 创建仿真、场景和机器人。
        super().__init__(cfg, render_mode, **kwargs)

        # 按照明确顺序查找关节，保证六维动作的含义固定。
        self.hip_joint_ids, self.hip_joint_names = self.robot.find_joints(
            ["left_hip_joint", "right_hip_joint"],
            preserve_order=True,
        )
        self.knee_joint_ids, self.knee_joint_names = self.robot.find_joints(
            ["left_knee_joint", "right_knee_joint"],
            preserve_order=True,
        )
        self.wheel_joint_ids, self.wheel_joint_names = self.robot.find_joints(
            ["left_wheel_joint", "right_wheel_joint"],
            preserve_order=True,
        )
        if len(self.hip_joint_ids) != 2:
            raise RuntimeError(f"髋关节数量错误：{self.hip_joint_names}")
        if len(self.knee_joint_ids) != 2:
            raise RuntimeError(f"膝关节数量错误：{self.knee_joint_names}")
        if len(self.wheel_joint_ids) != 2:
            raise RuntimeError(f"轮子关节数量错误：{self.wheel_joint_names}")

        # 四个腿关节的固定顺序为：
        # [左髋, 右髋, 左膝, 右膝]
        self.leg_joint_ids = self.hip_joint_ids + self.knee_joint_ids

        # 轮子允许接地；底盘、上腿和下腿接地属于非法接触。
        (
            self.illegal_contact_body_ids,
            self.illegal_contact_body_names,
        ) = self.contact_sensor.find_bodies(
            ["base_link", ".*upper_leg", ".*lower_leg"]
        )

        # 保存默认腿部姿态，观测使用相对于默认姿态的位置偏差。
        self.default_leg_pos = self.robot.data.default_joint_pos[
            :, self.leg_joint_ids
        ].clone()

        # 保存上一帧动作，用于动作变化率惩罚。
        self.previous_actions = torch.zeros_like(self.actions)

        # 六个关节最终写入仿真的实际力矩。
        self.joint_efforts = torch.zeros(
            (self.num_envs, self.robot.num_joints),
            dtype=torch.float32,
            device=self.device,
        )

        # 每行依次为 [前向速度, 偏航角速度, 车身高度]。
        self.commands = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

        # 每个并行环境独立记录距离下一次指令采样的剩余时间。
        self.command_time_left = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _setup_scene(self):
        # 创建机器人并注册到交互场景。
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 创建并注册接触传感器。
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # 创建地面并克隆并行环境。
        self.cfg.ground_cfg.func("/World/Ground", self.cfg.ground_cfg)
        self.scene.clone_environments(copy_from_source=False)

        # CPU仿真时显式过滤不同环境之间的碰撞。
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])

        # 添加环境光，方便GUI观察。
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _sample_commands(self, env_ids):
        # 没有需要更新的环境时直接返回。
        if env_ids.numel() == 0:
            return

        count = env_ids.numel()

        # 为指定环境分别采样前向速度、偏航角速度和车身高度。
        self.commands[env_ids, 0].uniform_(
            *self.cfg.command_forward_velocity_range
        )
        self.commands[env_ids, 1].uniform_(
            *self.cfg.command_yaw_velocity_range
        )
        self.commands[env_ids, 2].uniform_(
            *self.cfg.command_base_height_range
        )

        # 保留一定比例的原地指令，确保策略能学习零速度下稳定站立。
        # 此处只清零速度和转向，高度指令仍然有效，因此也会训练
        # 原地伸腿和原地缩腿。
        standing = (
            torch.rand(count, device=self.device)
            < self.cfg.standing_command_probability
        )
        self.commands[env_ids[standing], 0:2] = 0.0

        # 重置这些环境的指令倒计时。
        self.command_time_left[env_ids] = self.cfg.command_resampling_time_s

    def _pre_physics_step(self, actions):
        # 先保存旧动作，奖励函数需要计算相邻动作的变化量。
        self.previous_actions.copy_(self.actions)

        # 将NaN和无穷值转换成安全动作，防止污染物理仿真。
        safe_actions = torch.nan_to_num(
            actions,
            nan=0.0,
            posinf=self.cfg.action_clip,
            neginf=-self.cfg.action_clip,
        )

        # 将策略动作限制到[-1, 1]。
        self.actions.copy_(
            torch.clamp(
                safe_actions,
                min=-self.cfg.action_clip,
                max=self.cfg.action_clip,
            )
        )

        # 每个策略步减少一次倒计时；到期环境立即采样新指令。
        self.command_time_left -= self.step_dt
        resample_ids = torch.nonzero(
            self.command_time_left <= 0.0, as_tuple=False
        ).squeeze(-1)
        self._sample_commands(resample_ids)

    def _apply_action(self):
        # 清空上一物理步缓存的关节力矩。
        self.joint_efforts.zero_()

        # 六个动作全部直接映射为关节力矩：
        # [左髋, 右髋, 左膝, 右膝, 左轮, 右轮]
        self.joint_efforts[:, self.hip_joint_ids] = (
            self.actions[:, 0:2] * self.cfg.leg_effort_scale
        )
        self.joint_efforts[:, self.knee_joint_ids] = (
            self.actions[:, 2:4] * self.cfg.leg_effort_scale
        )
        self.joint_efforts[:, self.wheel_joint_ids] = (
            self.actions[:, 4:6] * self.cfg.wheel_effort_scale
        )

        # 将六个关节的目标力矩写入机器人缓存。
        self.robot.set_joint_effort_target(self.joint_efforts)

    def _get_observations(self):
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        # 腿关节位置使用相对于默认姿态的偏差。
        leg_pos_error = joint_pos[:, self.leg_joint_ids] - self.default_leg_pos

        # 先构造19维机器人状态观测。
        state_obs = torch.cat(
            (
                # 3维：重力在车身坐标系中的投影。
                self.robot.data.projected_gravity_b
                * self.cfg.gravity_obs_scale,
                # 3维：车身角速度。
                self.robot.data.root_ang_vel_b
                * self.cfg.angular_velocity_obs_scale,
                # 3维：车身线速度。
                self.robot.data.root_lin_vel_b
                * self.cfg.linear_velocity_obs_scale,
                # 4维：腿关节相对默认位置的偏差。
                leg_pos_error * self.cfg.leg_position_obs_scale,
                # 4维：腿关节速度。
                joint_vel[:, self.leg_joint_ids]
                * self.cfg.leg_velocity_obs_scale,
                # 2维：轮子速度。
                joint_vel[:, self.wheel_joint_ids]
                * self.cfg.wheel_velocity_obs_scale,
            ),
            dim=-1,
        )

        # 新增3维指令观测：前向速度、偏航角速度、目标高度偏差。
        # 高度减去默认高度后再缩放，使输入以零附近为中心。
        command_obs = torch.stack(
            (
                self.commands[:, 0] * self.cfg.command_forward_obs_scale,
                self.commands[:, 1] * self.cfg.command_yaw_obs_scale,
                (self.commands[:, 2] - self.cfg.target_base_height)
                * self.cfg.command_height_obs_scale,
            ),
            dim=-1,
        )

        # 最终顺序为：19维机器人状态、3维指令、6维刚刚执行的动作。
        return {
            "policy": torch.cat(
                (state_obs, command_obs, self.actions),
                dim=-1,
            )
        }

    def _get_dones(self):
        # 当前车身高度相对于所属并行环境原点。
        base_height = (
            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )

        # 高度过低通常表示机器人已经倒下或严重缩腿失稳。
        too_low = base_height < self.cfg.minimum_base_height

        # 高度异常升高时终止，防止无效仿真状态继续训练。
        too_high = base_height > self.cfg.maximum_base_height

        # 正常直立时projected_gravity_b的z约为-1；数值增大表示倾斜。
        too_tilted = (
            self.robot.data.projected_gravity_b[:, 2]
            > self.cfg.termination_gravity_z
        )

        # 检查底盘、上腿和下腿在最近几个物理步中的接触力。
        contact_forces = self.contact_sensor.data.net_forces_w_history
        illegal_contact = torch.any(
            torch.max(
                torch.norm(
                    contact_forces[:, :, self.illegal_contact_body_ids],
                    dim=-1,
                ),
                dim=1,
            ).values
            > self.cfg.illegal_contact_force_threshold,
            dim=1,
        )

        # 检查仿真状态是否出现NaN或无穷大。
        invalid_state = ~torch.isfinite(self.robot.data.root_state_w).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        invalid_state |= ~torch.isfinite(self.robot.data.joint_vel).all(dim=1)

        terminated = too_low | too_high | too_tilted | invalid_state
        terminated |= illegal_contact

        # episode_length_buf按策略步计数。
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self):
        # 所有速度均使用车身坐标系，前进方向固定为机体X轴。
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        base_height = (
            self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )
        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_ids]
        leg_vel = self.robot.data.joint_vel[:, self.leg_joint_ids]

        # 前向速度越接近指令，奖励越接近1。
        forward_tracking = torch.exp(
            -self.cfg.forward_tracking_error_scale
            * (lin_vel[:, 0] - self.commands[:, 0]).square()
        )

        # 绕车身Z轴的角速度越接近偏航指令，奖励越接近1。
        yaw_tracking = torch.exp(
            -self.cfg.yaw_tracking_error_scale
            * (ang_vel[:, 2] - self.commands[:, 1]).square()
        )

        # 车身高度越接近伸缩腿指令，奖励越接近1。
        height_tracking = torch.exp(
            -self.cfg.height_tracking_error_scale
            * (base_height - self.commands[:, 2]).square()
        )

        # 完全直立时gravity_z约为-1，upright约为1。
        upright = torch.clamp(-gravity_z, 0.0, 1.0).square()

        # 未因失败条件终止时获得存活奖励。
        alive = (~self.reset_terminated).float()

        # 惩罚车身上下弹跳。
        vertical_velocity_error = lin_vel[:, 2].square()

        # 只惩罚横滚和俯仰角速度，不惩罚任务需要的偏航角速度。
        roll_pitch_velocity_error = torch.sum(ang_vel[:, 0:2].square(), dim=1)

        # 只惩罚腿关节速度，不限制轮子为移动和平衡所需的转速。
        leg_velocity_error = torch.sum(leg_vel.square(), dim=1)

        # 惩罚过大的关节力矩指令。
        action_error = torch.sum(self.actions.square(), dim=1)

        # 惩罚相邻策略步之间过快的力矩变化。
        action_rate_error = torch.sum(
            (self.actions - self.previous_actions).square(), dim=1
        )

        # 关节超过软限位时进行惩罚。
        leg_limits = self.robot.data.soft_joint_pos_limits[:, self.leg_joint_ids]
        lower_error = torch.clamp(leg_limits[:, :, 0] - leg_pos, min=0.0)
        upper_error = torch.clamp(leg_pos - leg_limits[:, :, 1], min=0.0)
        joint_limit_error = torch.sum(
            lower_error.square() + upper_error.square(), dim=1
        )

        # 普通奖励按step_dt缩放，减少策略频率变化对总奖励量级的影响。
        rewards = {
            "forward_tracking": forward_tracking
            * self.cfg.reward_forward_tracking
            * self.step_dt,
            "yaw_tracking": yaw_tracking
            * self.cfg.reward_yaw_tracking
            * self.step_dt,
            "height_tracking": height_tracking
            * self.cfg.reward_height_tracking
            * self.step_dt,
            "upright": upright * self.cfg.reward_upright * self.step_dt,
            "alive": alive * self.cfg.reward_alive * self.step_dt,
            "vertical_velocity": vertical_velocity_error
            * self.cfg.penalty_vertical_velocity
            * self.step_dt,
            "roll_pitch_velocity": roll_pitch_velocity_error
            * self.cfg.penalty_roll_pitch_velocity
            * self.step_dt,
            "leg_velocity": leg_velocity_error
            * self.cfg.penalty_leg_velocity
            * self.step_dt,
            "action": action_error * self.cfg.penalty_action * self.step_dt,
            "action_rate": action_rate_error
            * self.cfg.penalty_action_rate
            * self.step_dt,
            "joint_limit": joint_limit_error
            * self.cfg.penalty_joint_limit
            * self.step_dt,
            # 倒下是瞬时事件，不乘step_dt。
            "termination": self.reset_terminated.float()
            * self.cfg.penalty_termination,
        }

        # 汇总所有奖励项，并阻止异常值传入训练器。
        total = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return torch.nan_to_num(
            total,
            nan=self.cfg.penalty_termination,
            posinf=self.cfg.penalty_termination,
            neginf=self.cfg.penalty_termination,
        )

    def _reset_idx(self, env_ids):
        # env_ids为None时重置全部并行环境。
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )

        # 重置Isaac Lab维护的场景缓存和episode计数器。
        super()._reset_idx(env_ids)

        # 清空动作历史和力矩缓存。
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.joint_efforts[env_ids] = 0.0

        # 将零力矩写入仿真，防止重置前的力矩残留。
        self.robot.set_joint_effort_target(
            self.joint_efforts[env_ids], env_ids=env_ids
        )

        # 复制默认根状态：[位置3, 四元数4, 线速度3, 角速度3]。
        root_state = self.robot.data.default_root_state[env_ids].clone()

        # 加上各个并行环境在世界坐标系中的原点。
        root_state[:, 0:3] += self.scene.env_origins[env_ids]
        count = env_ids.numel()

        # 为每台重置机器人随机生成一个很小的初始俯仰角。
        initial_pitch = torch.empty(
            count, dtype=torch.float32, device=self.device
        ).uniform_(
            -self.cfg.initial_pitch_range,
            self.cfg.initial_pitch_range,
        )
        zero_angle = torch.zeros_like(initial_pitch)

        # Isaac Lab四元数顺序为[w, x, y, z]。
        pitch_quaternion = quat_from_euler_xyz(
            zero_angle, initial_pitch, zero_angle
        )
        root_state[:, 3:7] = quat_mul(
            root_state[:, 3:7], pitch_quaternion
        )

        # 复制默认关节位置和速度。
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 给四个腿关节添加少量初始位置扰动。
        joint_pos[:, self.leg_joint_ids] += torch.empty(
            (count, len(self.leg_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(
            -self.cfg.initial_leg_position_noise,
            self.cfg.initial_leg_position_noise,
        )

        # 给全部六个关节添加少量初始速度扰动。
        joint_vel += torch.empty_like(joint_vel).uniform_(
            -self.cfg.initial_joint_velocity_noise,
            self.cfg.initial_joint_velocity_noise,
        )

        # 防止随机化后的腿关节位置超过软限位。
        leg_limits = self.robot.data.soft_joint_pos_limits[env_ids][
            :, self.leg_joint_ids
        ]
        leg_pos = joint_pos[:, self.leg_joint_ids]
        joint_pos[:, self.leg_joint_ids] = torch.maximum(
            torch.minimum(leg_pos, leg_limits[:, :, 1]),
            leg_limits[:, :, 0],
        )

        # 将根姿态、根速度和关节状态写回仿真。
        self.robot.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(
            root_state[:, 7:13], env_ids=env_ids
        )
        self.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids
        )

        # 为新episode采样第一条移动、转向和高度指令。
        self._sample_commands(env_ids)
