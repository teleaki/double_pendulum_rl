import torch


class SustainedThresholdPenalty:
    """让误差惩罚倍率随连续超阈值时间增长。

    每个并行环境维护一个独立计时器。误差大于 ``tolerance`` 时，
    连续超差时间按 ``step_dt`` 累加；误差回到阈值以内时立即清零。
    ``update()`` 返回的倍率最小为1，表示普通惩罚；连续超差后倍率为：

        1 + elapsed_time * growth_per_second

    累计时间由 ``max_accumulation_time`` 限制，避免惩罚无限增长。
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        step_dt: float,
        tolerance: float,
        growth_per_second: float = 1.0,
        max_accumulation_time: float = 5.0,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if step_dt <= 0.0:
            raise ValueError("step_dt must be positive.")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        if growth_per_second < 0.0:
            raise ValueError("growth_per_second must be non-negative.")
        if max_accumulation_time <= 0.0:
            raise ValueError("max_accumulation_time must be positive.")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = float(step_dt)
        self.tolerance = float(tolerance)
        self.growth_per_second = float(growth_per_second)
        self.max_accumulation_time = float(max_accumulation_time)

        self.elapsed_time = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def update(self, error_magnitude: torch.Tensor) -> torch.Tensor:
        """更新连续超差时间并返回当前惩罚倍率。"""

        if error_magnitude.shape != (self.num_envs,):
            raise ValueError(
                "error_magnitude must have shape "
                f"({self.num_envs},), got {tuple(error_magnitude.shape)}."
            )
        if error_magnitude.device != self.elapsed_time.device:
            raise ValueError(
                "error_magnitude and elapsed_time must be on the same device."
            )

        self.elapsed_time.copy_(
            torch.where(
                error_magnitude > self.tolerance,
                torch.clamp(
                    self.elapsed_time + self.step_dt,
                    max=self.max_accumulation_time,
                ),
                torch.zeros_like(self.elapsed_time),
            )
        )

        return 1.0 + self.elapsed_time * self.growth_per_second

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """清零全部环境或指定环境的连续超差时间。"""

        if env_ids is None:
            self.elapsed_time.zero_()
            return
        self.elapsed_time[env_ids] = 0.0
