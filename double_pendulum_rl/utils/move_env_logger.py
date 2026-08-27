import torch

from double_pendulum_rl.utils.sustained_threshold_penalty import (
    SustainedThresholdPenalty,
)


class MoveEnvLogger:
    """集中生成MoveEnv写入训练器的诊断指标。

    这个类只读取环境已经计算好的张量，不修改环境状态，也不参与奖励
    计算。这样可以让环境的奖励函数专注于任务逻辑，同时保持原有日志名
    不变，已有TensorBoard曲线仍可继续对照。
    """

    def __init__(
        self,
        cfg,
        roll_pitch_penalty: SustainedThresholdPenalty,
        forward_velocity_penalty: SustainedThresholdPenalty,
        yaw_velocity_penalty: SustainedThresholdPenalty,
        leg_length_penalty: SustainedThresholdPenalty,
    ):
        self.cfg = cfg
        self.roll_pitch_penalty = roll_pitch_penalty
        self.forward_velocity_penalty = forward_velocity_penalty
        self.yaw_velocity_penalty = yaw_velocity_penalty
        self.leg_length_penalty = leg_length_penalty

    def build(
        self,
        *,
        commands: torch.Tensor,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        average_leg_length: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        forward_velocity_error: torch.Tensor,
        yaw_velocity_error: torch.Tensor,
        leg_length_error: torch.Tensor,
        roll_pitch_angle_error: torch.Tensor,
        actions: torch.Tensor,
        knee_flexion_limit_error: torch.Tensor,
        knee_extension_limit_error: torch.Tensor,
        reset_terminated: torch.Tensor,
        termination_reasons: dict[str, torch.Tensor],
        rewards: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """根据当前策略步的数据生成标量训练日志。"""

        forward_abs_error = torch.abs(forward_velocity_error)
        yaw_abs_error = torch.abs(yaw_velocity_error)
        leg_length_abs_error = torch.sqrt(leg_length_error)
        roll_pitch_magnitude = torch.sqrt(roll_pitch_angle_error)

        forward_success = (
            forward_abs_error <= self.cfg.forward_velocity_error_tolerance
        )
        yaw_success = yaw_abs_error <= self.cfg.yaw_velocity_error_tolerance
        leg_length_success = (
            leg_length_abs_error <= self.cfg.leg_length_error_tolerance
        )
        orientation_success = (
            roll_pitch_magnitude <= self.cfg.roll_pitch_error_tolerance
        )

        logs = {
            "Command/forward_abs_mean": torch.mean(torch.abs(commands[:, 0])),
            "Command/forward_std": torch.std(commands[:, 0], unbiased=False),
            "Command/yaw_abs_mean": torch.mean(torch.abs(commands[:, 1])),
            "Command/yaw_std": torch.std(commands[:, 1], unbiased=False),
            "Command/leg_length_mean": torch.mean(commands[:, 2]),
            "Command/leg_length_std": torch.std(commands[:, 2], unbiased=False),
            "State/forward_velocity_abs_mean": torch.mean(
                torch.abs(linear_velocity[:, 0])
            ),
            "State/yaw_velocity_abs_mean": torch.mean(
                torch.abs(angular_velocity[:, 2])
            ),
            "State/average_leg_length": torch.mean(average_leg_length),
            "State/roll_abs_deg": torch.rad2deg(torch.mean(torch.abs(roll))),
            "State/pitch_abs_deg": torch.rad2deg(torch.mean(torch.abs(pitch))),
            "Tracking/forward_abs_error": torch.mean(forward_abs_error),
            "Tracking/yaw_abs_error": torch.mean(yaw_abs_error),
            "Tracking/leg_length_abs_error": torch.mean(leg_length_abs_error),
            "Tracking/roll_pitch_abs_deg": torch.rad2deg(
                torch.mean(roll_pitch_magnitude)
            ),
            "Success/forward": torch.mean(forward_success.float()),
            "Success/yaw": torch.mean(yaw_success.float()),
            "Success/leg_length": torch.mean(leg_length_success.float()),
            "Success/flat_orientation": torch.mean(orientation_success.float()),
            "Success/all_commands": torch.mean(
                (forward_success & yaw_success & leg_length_success).float()
            ),
            "Success/all_with_orientation": torch.mean(
                (
                    forward_success
                    & yaw_success
                    & leg_length_success
                    & orientation_success
                ).float()
            ),
            "ErrorTime/forward": torch.mean(
                self.forward_velocity_penalty.elapsed_time
            ),
            "ErrorTime/yaw": torch.mean(self.yaw_velocity_penalty.elapsed_time),
            "ErrorTime/leg_length": torch.mean(
                self.leg_length_penalty.elapsed_time
            ),
            "ErrorTime/roll_pitch": torch.mean(
                self.roll_pitch_penalty.elapsed_time
            ),
            "Action/leg_abs_mean": torch.mean(torch.abs(actions[:, 0:4])),
            "Action/wheel_abs_mean": torch.mean(torch.abs(actions[:, 4:6])),
            "Joint/knee_flexion_limit_rate": torch.mean(
                (knee_flexion_limit_error > 0.0).float()
            ),
            "Joint/knee_extension_limit_rate": torch.mean(
                (knee_extension_limit_error > 0.0).float()
            ),
            "Termination/rate": torch.mean(reset_terminated.float()),
        }

        # 分项终止率和奖励便于定位具体失败原因及奖励量级问题。
        for reason_name, reason_mask in termination_reasons.items():
            logs[f"Termination/{reason_name}"] = torch.mean(reason_mask.float())
        for reward_name, reward_value in rewards.items():
            logs[f"Reward/{reward_name}"] = torch.mean(reward_value)

        return logs
