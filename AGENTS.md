# AGENTS.md

## Purpose

This repository is an Isaac Lab reinforcement-learning project for a wheel-legged robot.

This file defines the architectural rules that coding agents must follow when creating, modifying, testing, training, or evaluating RL tasks.

The central design principle is:

> **Reuse the project interface and environment lifecycle.
> Change task configuration and reward logic.**

A new RL task is normally **not** a new environment architecture.

A new task should usually reuse:

* the same robot
* the same action interface
* the same observation interface
* the same command interface
* the same scene structure
* the same action-processing pipeline
* the same reset structure
* the same generic scripts

The largest task-specific differences should normally be:

* command ranges
* command curriculum
* reward parameters
* `_get_rewards()`
* occasionally termination/reset details

---

# 1. Architecture Contract

The project follows this dependency structure:

```text
URDF
  ↓
robot.py
  ↓
task environment
  ↓
task PPO configuration
  ↓
Gym registration
  ↓
generic smoke-test / train / play scripts
```

More concretely:

```text
assets/
   ↓
double_pendulum_rl/robot.py
   ↓
double_pendulum_rl/env/<task>_env.py
   ↓
double_pendulum_rl/agents/<task>_ppo_cfg.py
   ↓
double_pendulum_rl/__init__.py
   ↓
scripts/
```

Dependencies must remain one-way.

Do not introduce task-specific knowledge into generic infrastructure.

---

# 2. Repository Structure

```text
.
├── assets/
│   └── urdf/
│       └── ...
│
├── double_pendulum_rl/
│   ├── __init__.py
│   ├── robot.py
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   └── <task>_env.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── <task>_ppo_cfg.py
│   │
│   └── utils/
│       └── ...
│
└── scripts/
    ├── smoke_test.py
    ├── train.py
    ├── play.py
    ├── test_joint.py
    └── view_robot.py
```

For a normal new task, expected changes are:

```text
ADD:
double_pendulum_rl/env/<task>_env.py

ADD:
double_pendulum_rl/agents/<task>_ppo_cfg.py

EDIT:
double_pendulum_rl/__init__.py
```

Reusable helper logic may additionally be added to:

```text
double_pendulum_rl/utils/
```

A normal task should not require task-specific implementation changes to:

```text
scripts/smoke_test.py
scripts/train.py
scripts/play.py
```

Selecting which registered task to run may be done by editing the clearly
marked parameter block at the top of these scripts.

---

# 3. Robot Description

## `assets/`

`assets/` contains robot model files, primarily URDF.

The URDF is the source of truth for physical robot structure such as:

* links
* joints
* joint names
* joint axes
* joint limits
* visual geometry
* collision geometry
* inertial properties

If the physical robot changes, modify the URDF first.

Task environments must not redefine the robot structure independently.

---

# 4. Shared Robot Configuration

## `double_pendulum_rl/robot.py`

`robot.py` converts the URDF into the Isaac Lab `ArticulationCfg` shared by all RL tasks.

It is responsible for robot-level configuration such as:

* locating the URDF
* loading the URDF
* fixed/floating base configuration
* initial root pose
* initial joint positions
* initial joint velocities
* actuator grouping
* actuator effort limits
* actuator velocity limits
* actuator stiffness
* actuator damping
* robot-level contact sensor activation

Task environments must reuse:

```python
ROBOT_CFG
```

Typical usage:

```python
robot_cfg = ROBOT_CFG.replace(
    prim_path="/World/envs/env_.*/Robot"
)
```

Do not duplicate the complete articulation configuration inside individual tasks.

`robot.py` describes the robot.

`<task>_env.py` describes the RL task.

---

# 5. Project-Wide RL Interface

All normal RL tasks must use the same policy interface.

The project-wide interface is:

```text
action dimension      = 6
observation dimension = 28
command dimension     = 3
```

These are project-level contracts.

A normal task must not redefine them independently.

---

# 6. Fixed Action Interface

The action vector is always:

```text
[
    left_hip,
    right_hip,
    left_knee,
    right_knee,
    left_wheel,
    right_wheel,
]
```

Therefore:

```python
action_space = 6
```

The meaning of every action index is fixed:

```text
action[0] = left hip command
action[1] = right hip command
action[2] = left knee command
action[3] = right knee command
action[4] = left wheel command
action[5] = right wheel command
```

The current control interface is normalized torque control:

```text
policy output
    ↓
sanitize
    ↓
clip to [-1, 1]
    ↓
scale by configured effort
    ↓
joint torque command
```

Leg actions use:

```python
leg_effort_scale
```

Wheel actions use:

```python
wheel_effort_scale
```

The action ordering and semantics must not change between normal tasks.

Changing the action interface is a project-wide interface change, not a task-level tuning change.

---

# 7. Fixed Command Interface

All normal tasks use a three-dimensional command:

```text
[
    target_forward_velocity,
    target_yaw_velocity,
    target_leg_length,
]
```

Therefore:

```text
command dimension = 3
```

The meaning is fixed:

```text
commands[:, 0]
    target forward velocity in robot body X direction

commands[:, 1]
    target yaw angular velocity around robot body Z axis

commands[:, 2]
    target leg length
```

Different tasks should normally differ by command ranges rather than command definitions.

If a task does not vary one command component, preserve the component and use a fixed range.

For example:

```python
command_forward_velocity_range = (0.0, 0.0)
command_yaw_velocity_range = (0.0, 0.0)
command_leg_length_range = (0.35, 0.35)
```

Do not remove command dimensions merely because a task does not actively vary them.

---

# 8. Fixed Observation Interface

All normal tasks use the same 28-dimensional policy observation.

The observation layout is:

```text
3  projected gravity in body frame
3  body angular velocity
3  body linear velocity
4  leg joint position error
4  leg joint velocity
2  wheel joint velocity
3  command
6  current action
-------------------------
28 dimensions
```

The ordering must remain exactly:

```text
[
    projected_gravity_b,          # 3
    root_ang_vel_b,               # 3
    root_lin_vel_b,               # 3
    leg_joint_position_error,     # 4
    leg_joint_velocity,           # 4
    wheel_joint_velocity,         # 2
    command_observation,          # 3
    action,                       # 6
]
```

Therefore:

```python
observation_space = 28
```

Normal tasks must not add, remove, or reorder observations independently.

If a new signal is truly required across the project, treat that as a deliberate project-wide policy-interface change.

---

# 9. Observation Meaning

## Projected gravity

Three dimensions:

```python
self.robot.data.projected_gravity_b
```

This provides orientation information in the robot body frame.

---

## Body angular velocity

Three dimensions:

```python
self.robot.data.root_ang_vel_b
```

Use body-frame angular velocity.

---

## Body linear velocity

Three dimensions:

```python
self.robot.data.root_lin_vel_b
```

Use body-frame linear velocity.

---

## Leg joint position error

Four dimensions with fixed ordering:

```text
[
    left_hip,
    right_hip,
    left_knee,
    right_knee,
]
```

Use position relative to the default joint state:

```python
joint_pos[:, self.leg_joint_ids] - self.default_leg_pos
```

---

## Leg joint velocity

Four dimensions with the same ordering:

```text
[
    left_hip,
    right_hip,
    left_knee,
    right_knee,
]
```

---

## Wheel joint velocity

Two dimensions:

```text
[
    left_wheel,
    right_wheel,
]
```

---

## Command observation

Three dimensions corresponding to the fixed command interface:

```text
[
    target_forward_velocity,
    target_yaw_velocity,
    target_leg_length,
]
```

Command values may be scaled before entering the policy.

Leg length may be represented relative to a configured center value.

The command meaning and ordering must remain fixed.

---

## Action history

Six dimensions.

Use:

```python
self.actions
```

with the same fixed action ordering.

---

# 10. EnvCfg Organization

Every environment must define:

```python
@configclass
class <Task>EnvCfg(DirectRLEnvCfg):
    ...
```

`EnvCfg` must be conceptually divided into two major sections:

```text
FIXED INTERFACE CONFIGURATION
        +
TASK CONFIGURATION
```

The distinction must remain clear in code.

---

# 11. Fixed Interface Configuration

The fixed-interface section defines the shared robot-policy contract and common environment structure.

Normal tasks should copy this section from the project-standard environment implementation and leave it unchanged unless there is a deliberate project-wide interface modification.

Recommended organization:

```python
@configclass
class <Task>EnvCfg(DirectRLEnvCfg):

    # =========================================================
    # Fixed interface configuration
    # =========================================================

    # Time / simulation structure
    decimation = 2

    # Policy interface
    action_space = 6
    observation_space = 28

    # Simulation
    sim = SimulationCfg(
        ...
    )

    # Scene
    scene = InteractiveSceneCfg(
        ...
    )

    # Shared robot
    robot_cfg = ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Ground / sensors
    ground_cfg = ...
    contact_sensor_cfg = ...

    # Action interface
    action_clip = 1.0
    leg_effort_scale = ...
    wheel_effort_scale = ...

    # Observation scaling
    gravity_obs_scale = ...
    angular_velocity_obs_scale = ...
    linear_velocity_obs_scale = ...
    leg_position_obs_scale = ...
    leg_velocity_obs_scale = ...
    wheel_velocity_obs_scale = ...

    # Command observation scaling
    command_forward_obs_scale = ...
    command_yaw_obs_scale = ...
    command_leg_length_obs_scale = ...
    command_leg_length_center = ...

    # Shared robot geometry if required by common logic
    upper_leg_length = ...
    lower_leg_length = ...


    # =========================================================
    # Task configuration
    # =========================================================

    ...
```

---

# 12. What Is Fixed

The following should normally be treated as project-wide interface or common-environment configuration:

```text
action_space
observation_space

action ordering
observation ordering
command ordering

action semantics
observation semantics
command semantics

joint grouping
joint lookup semantics

action clipping convention
action-to-joint mapping

observation construction

robot configuration

basic simulation structure
basic scene structure
basic contact sensor setup

observation scaling convention
```

These should not be changed merely to make a particular task easier to train.

---

# 13. Task Configuration

The task configuration defines the RL problem rather than the policy interface.

Typical sections include:

```text
episode duration

command ranges
command sampling distribution
command resampling period
command curriculum

reset randomization

termination thresholds

reward weights
reward error scales
reward tolerances

task-specific helper parameters
```

Example:

```python
# =========================================================
# Task configuration
# =========================================================

# Episode
episode_length_s = ...

# Commands
command_forward_velocity_range = (...)
command_yaw_velocity_range = (...)
command_leg_length_range = (...)
command_resampling_time_s = ...

# Reset randomization
initial_pitch_range = ...
initial_leg_position_noise = ...
initial_joint_velocity_noise = ...

# Termination
minimum_base_height = ...
maximum_base_height = ...
termination_gravity_z = ...
illegal_contact_force_threshold = ...

# Rewards
reward_xxx = ...
penalty_xxx = ...
reward_error_scale_xxx = ...

# Curriculum / task-specific parameters
...
```

This is the primary configuration area agents are expected to modify when creating a new task.

---

# 14. Standard Environment Structure

All normal environments should follow:

```python
@configclass
class <Task>EnvCfg(DirectRLEnvCfg):
    ...


class <Task>Env(DirectRLEnv):
    cfg: <Task>EnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        ...

    def _setup_scene(self):
        ...

    def _sample_commands(self, env_ids):
        ...

    def _pre_physics_step(self, actions):
        ...

    def _apply_action(self):
        ...

    def _get_observations(self):
        ...

    def _get_dones(self):
        ...

    def _get_rewards(self):
        ...

    def _reset_idx(self, env_ids):
        ...
```

For most future tasks:

```text
__init__
_setup_scene
_sample_commands
_pre_physics_step
_apply_action
_get_observations
_get_dones
_reset_idx
```

should require only small adaptations.

The largest task-specific implementation is expected to be:

```python
_get_rewards()
```

---

# 15. Expected Amount of Task-Specific Change

Use this mental model:

```text
__init__                mostly fixed
_setup_scene            mostly fixed
_sample_commands        mostly config-driven
_pre_physics_step       almost fixed
_apply_action           almost fixed
_get_observations       almost fixed
_get_dones              mostly fixed
_get_rewards            PRIMARY TASK LOGIC
_reset_idx              mostly fixed
```

Do not rewrite stable methods unnecessarily.

A new task should usually modify configuration before modifying environment lifecycle code.

---

# 16. `__init__()`

`__init__()` establishes runtime state.

Typical shared responsibilities:

* resolve hip joint IDs
* resolve knee joint IDs
* resolve wheel joint IDs
* validate joint counts
* build `leg_joint_ids`
* resolve illegal-contact bodies
* cache default leg position
* allocate previous-action buffer
* allocate effort buffer
* allocate command tensor
* allocate command timer
* allocate command-resample state
* construct helper classes
* construct logger if needed

Joint lookup must use names.

Example:

```python
self.hip_joint_ids, self.hip_joint_names = self.robot.find_joints(
    [
        "left_hip_joint",
        "right_hip_joint",
    ],
    preserve_order=True,
)
```

Use:

```python
preserve_order=True
```

whenever action or observation semantics depend on order.

Validate expected joint counts immediately.

Do not hard-code simulator joint indices.

---

# 17. Standard Joint Ordering

The fixed leg joint order is:

```text
[
    left_hip,
    right_hip,
    left_knee,
    right_knee,
]
```

Use:

```python
self.leg_joint_ids = self.hip_joint_ids + self.knee_joint_ids
```

The wheel order is:

```text
[
    left_wheel,
    right_wheel,
]
```

These orderings are part of the project-wide policy interface.

---

# 18. `_setup_scene()`

`_setup_scene()` should be nearly identical across normal tasks.

Its responsibilities are:

1. create the robot
2. register the robot
3. create contact sensors
4. register sensors
5. create ground or terrain
6. clone environments
7. configure collision filtering
8. create visualization lighting

Typical structure:

```python
def _setup_scene(self):
    self.robot = Articulation(self.cfg.robot_cfg)
    self.scene.articulations["robot"] = self.robot

    self.contact_sensor = ContactSensor(
        self.cfg.contact_sensor_cfg
    )
    self.scene.sensors["contact_sensor"] = self.contact_sensor

    self.cfg.ground_cfg.func(
        "/World/Ground",
        self.cfg.ground_cfg,
    )

    self.scene.clone_environments(
        copy_from_source=False
    )

    if self.device == "cpu":
        self.scene.filter_collisions(
            global_prim_paths=[
                "/World/Ground"
            ]
        )

    ...
```

Do not modify `_setup_scene()` for reward design.

Only change it when the task truly requires different simulation entities, sensors, terrain, or objects.

---

# 19. Command State

All normal environments should maintain:

```python
self.commands
```

with shape:

```text
(num_envs, 3)
```

The command meaning is always:

```text
column 0 = target forward velocity
column 1 = target yaw velocity
column 2 = target leg length
```

Normal environments should also maintain:

```python
self.command_time_left
```

with shape:

```text
(num_envs,)
```

If delayed command switching is used, maintain:

```python
self.command_resample_pending
```

with shape:

```text
(num_envs,)
```

and boolean dtype.

---

# 20. `_sample_commands()`

`_sample_commands(env_ids)` is part of the standard environment lifecycle.

Its normal purpose is:

```text
sample commands
        ↓
write commands for env_ids
        ↓
reset command timers
        ↓
reset command-dependent helper state
```

The implementation should ideally remain mostly unchanged between tasks.

Task differences should primarily be expressed through:

```python
self.cfg.command_forward_velocity_range
self.cfg.command_yaw_velocity_range
self.cfg.command_leg_length_range
```

rather than completely different command-sampling implementations.

---

# 21. Standard Command Sampling

Preferred pattern:

```python
def _sample_commands(self, env_ids):
    if env_ids.numel() == 0:
        return

    count = env_ids.numel()

    sampled_commands = torch.empty(
        (count, 3),
        dtype=torch.float32,
        device=self.device,
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

    self.command_time_left[env_ids] = (
        self.cfg.command_resampling_time_s
    )
```

Task-specific command distributions may extend this method when genuinely necessary.

Prefer configuration changes before implementation changes.

---

# 22. Fixed Commands Through Ranges

A task requiring a constant target should normally use zero-width command ranges.

Example:

```python
command_forward_velocity_range = (0.0, 0.0)
command_yaw_velocity_range = (0.0, 0.0)
command_leg_length_range = (0.35, 0.35)
```

Do not remove the command interface.

This preserves:

```text
command dimension
observation dimension
observation ordering
policy interface
```

across tasks.

---

# 23. Vectorized Command Sampling

Command sampling must be vectorized.

Do not use:

```python
for env_id in env_ids:
    ...
```

for normal command generation.

Avoid relying on in-place advanced indexing such as:

```python
self.commands[env_ids, 0].uniform_(...)
```

Prefer sampling into a temporary tensor and writing it back:

```python
sampled_commands = torch.empty(
    (env_ids.numel(), 3),
    device=self.device,
)

...

self.commands[env_ids] = sampled_commands
```

---

# 24. Command Timing Consistency

The command used for reward calculation must be the command that generated the current action.

Never use:

```text
policy observes command A
        ↓
policy produces action
        ↓
command changes to B
        ↓
reward evaluates action against B
```

Use:

```text
policy observes command A
        ↓
policy produces action
        ↓
physics step
        ↓
reward evaluates action against A
        ↓
command changes to B
        ↓
next observation contains command B
```

The `command_resample_pending` pattern is appropriate for preserving this invariant.

---

# 25. `_pre_physics_step()`

`_pre_physics_step()` should normally remain almost identical across tasks.

Its shared responsibilities are:

```text
save previous action
        ↓
sanitize NaN / Inf
        ↓
clip normalized action
        ↓
store action
        ↓
update command timer
        ↓
mark expired commands
```

Typical structure:

```python
def _pre_physics_step(self, actions):
    self.previous_actions.copy_(
        self.actions
    )

    safe_actions = torch.nan_to_num(
        actions,
        nan=0.0,
        posinf=self.cfg.action_clip,
        neginf=-self.cfg.action_clip,
    )

    self.actions.copy_(
        torch.clamp(
            safe_actions,
            min=-self.cfg.action_clip,
            max=self.cfg.action_clip,
        )
    )

    self.command_time_left -= self.step_dt

    self.command_resample_pending |= (
        self.command_time_left <= 0.0
    )
```

Do not put reward logic here.

---

# 26. `_apply_action()`

`_apply_action()` should normally be identical across tasks.

The fixed mapping is:

```text
actions[:, 0:2] → hip joints
actions[:, 2:4] → knee joints
actions[:, 4:6] → wheel joints
```

Typical structure:

```python
def _apply_action(self):
    self.joint_efforts.zero_()

    self.joint_efforts[
        :,
        self.hip_joint_ids,
    ] = (
        self.actions[:, 0:2]
        * self.cfg.leg_effort_scale
    )

    self.joint_efforts[
        :,
        self.knee_joint_ids,
    ] = (
        self.actions[:, 2:4]
        * self.cfg.leg_effort_scale
    )

    self.joint_efforts[
        :,
        self.wheel_joint_ids,
    ] = (
        self.actions[:, 4:6]
        * self.cfg.wheel_effort_scale
    )

    self.robot.set_joint_effort_target(
        self.joint_efforts
    )
```

Do not modify action semantics for individual tasks.

---

# 27. `_get_observations()`

`_get_observations()` should normally remain almost identical across tasks.

It implements the fixed 28-dimensional policy interface.

Typical structure:

```python
def _get_observations(self):
    joint_pos = self.robot.data.joint_pos
    joint_vel = self.robot.data.joint_vel

    leg_pos_error = (
        joint_pos[:, self.leg_joint_ids]
        -
        self.default_leg_pos
    )

    state_obs = torch.cat(
        (
            self.robot.data.projected_gravity_b
            * self.cfg.gravity_obs_scale,

            self.robot.data.root_ang_vel_b
            * self.cfg.angular_velocity_obs_scale,

            self.robot.data.root_lin_vel_b
            * self.cfg.linear_velocity_obs_scale,

            leg_pos_error
            * self.cfg.leg_position_obs_scale,

            joint_vel[:, self.leg_joint_ids]
            * self.cfg.leg_velocity_obs_scale,

            joint_vel[:, self.wheel_joint_ids]
            * self.cfg.wheel_velocity_obs_scale,
        ),
        dim=-1,
    )

    command_obs = torch.stack(
        (
            self.commands[:, 0]
            * self.cfg.command_forward_obs_scale,

            self.commands[:, 1]
            * self.cfg.command_yaw_obs_scale,

            (
                self.commands[:, 2]
                -
                self.cfg.command_leg_length_center
            )
            * self.cfg.command_leg_length_obs_scale,
        ),
        dim=-1,
    )

    observations = torch.cat(
        (
            state_obs,
            command_obs,
            self.actions,
        ),
        dim=-1,
    )

    return {
        "policy": observations,
    }
```

Normal tasks should not redesign this layout.

---

# 28. Observation Interface Changes

Do not modify observation dimension or ordering merely because one task appears to benefit from another input.

Before changing the observation interface, determine:

```text
Can this quantity be derived from existing observations?

Is this quantity really necessary for deployment?

Should this information be available to every task?

Is this a task-level requirement or a project-wide interface requirement?
```

If the observation interface must change, treat it as a deliberate project-wide redesign.

---

# 29. `_get_dones()`

`_get_dones()` should remain mostly shared across tasks.

Common checks include:

* base too low
* base too high
* excessive tilt
* illegal contact
* invalid root state
* invalid joint position
* invalid joint velocity
* timeout

Task differences should preferably be expressed through thresholds in `EnvCfg`.

Example:

```python
minimum_base_height = ...
maximum_base_height = ...
termination_gravity_z = ...
illegal_contact_force_threshold = ...
```

Only modify `_get_dones()` logic when the task genuinely requires different failure semantics.

---

# 30. Numerical Termination Safety

Always detect non-finite simulator state.

Example:

```python
invalid_state = (
    ~torch.isfinite(
        self.robot.data.root_state_w
    ).all(dim=1)
)

invalid_state |= (
    ~torch.isfinite(
        self.robot.data.joint_pos
    ).all(dim=1)
)

invalid_state |= (
    ~torch.isfinite(
        self.robot.data.joint_vel
    ).all(dim=1)
)
```

Invalid environments should terminate and reset.

Do not allow NaN or Inf to propagate into PPO.

---

# 31. `_get_rewards()` Is the Primary Task Definition

`_get_rewards()` is expected to contain the largest task-specific differences.

This method defines:

> What behavior should the policy learn?

All normal tasks use the same:

```text
robot
action interface
observation interface
command interface
simulation lifecycle
```

The main difference is how behavior is rewarded.

Coding agents should focus most task-specific design work here.

---

# 32. Reward Design Process

Before implementing rewards, identify:

1. the command
2. the physical behavior expected from that command
3. the tracking error
4. the primary reward
5. necessary safety or regularization terms

Conceptually:

```text
command
    ↓
desired behavior
    ↓
actual behavior
    ↓
tracking error
    ↓
reward
```

The primary reward should directly represent task success.

---

# 33. Reward Structure

Prefer named reward terms.

Example:

```python
rewards = {
    "tracking_a": ...,
    "tracking_b": ...,
    "stability": ...,
    "action": ...,
    "action_rate": ...,
    "termination": ...,
}
```

Then:

```python
total_reward = torch.sum(
    torch.stack(
        list(rewards.values())
    ),
    dim=0,
)
```

Finally:

```python
return torch.nan_to_num(
    total_reward,
    ...
)
```

Named rewards make logging and diagnosis easier.

---

# 34. Reward Configuration

Reward weights and shaping parameters belong in the task-configuration section of `EnvCfg`.

Prefer:

```python
tracking_reward
* self.cfg.reward_tracking
```

instead of:

```python
tracking_reward * 2.0
```

Task configuration may include:

```text
reward weights
penalty weights
tracking error scales
tracking tolerances
temporal penalty parameters
joint safety margins
task-specific geometry parameters
```

These are expected to vary significantly between tasks.

---

# 35. Continuous Reward Scaling

Continuous rewards and penalties should normally be multiplied by:

```python
self.step_dt
```

Example:

```python
reward = (
    reward_term
    * self.cfg.reward_weight
    * self.step_dt
)
```

This reduces sensitivity of total reward magnitude to policy frequency.

One-time events such as termination penalties should normally not be multiplied by `step_dt`.

---

# 36. Avoid Reward Over-Engineering

Do not automatically copy every reward or penalty from another task.

Task-specific shaping may include:

* symmetry penalties
* sustained tracking penalties
* alignment penalties
* specialized joint-limit penalties
* task-specific kinematics
* curriculum shaping
* special command distributions

These are not project-wide defaults.

Use the process:

```text
simple reward
        ↓
train / diagnose
        ↓
identify failure mode
        ↓
add targeted shaping if needed
```

rather than:

```text
copy all existing reward terms
        ↓
hope training works
```

---

# 37. Command Resampling After Reward

If commands expire during a step, resampling should happen only after the current reward has been evaluated.

Typical sequence:

```python
resample_mask = (
    self.command_resample_pending
    &
    ~self.reset_terminated
)

resample_ids = torch.nonzero(
    resample_mask,
    as_tuple=False,
).squeeze(-1)

self.command_resample_pending.zero_()

self._sample_commands(
    resample_ids
)
```

The next observation then contains the new command.

---

# 38. `_reset_idx()`

`_reset_idx()` should remain mostly identical across normal tasks.

Typical sequence:

```text
normalize env_ids
        ↓
super()._reset_idx(env_ids)
        ↓
clear actions
        ↓
clear previous actions
        ↓
clear effort buffers
        ↓
clear command-resample flags
        ↓
reset temporal helpers
        ↓
copy default root state
        ↓
offset by environment origin
        ↓
randomize initial orientation
        ↓
copy default joint state
        ↓
randomize joint state
        ↓
clamp to joint limits
        ↓
write state to simulator
        ↓
sample initial command
```

A normal reset should end with:

```python
self._sample_commands(env_ids)
```

---

# 39. Reset Randomization

Task-specific reset randomization should normally be controlled by `EnvCfg`.

Examples:

```python
initial_pitch_range = ...
initial_leg_position_noise = ...
initial_joint_velocity_noise = ...
```

Prefer changing these values rather than rewriting reset structure.

---

# 40. Temporal State Reset

Every tensor or helper carrying information across policy steps must be considered during reset.

Examples:

```text
previous actions
command timers
pending command switches
error duration
tracking duration
integrators
curriculum state
stateful penalties
```

A new episode must not unintentionally inherit state from the previous episode.

Stateful helpers should provide:

```python
reset(env_ids)
```

when appropriate.

---

# 41. `utils/`

`double_pendulum_rl/utils/` is for reusable or independently meaningful logic.

Good candidates include:

* reusable reward helpers
* stateful penalty helpers
* command sampling helpers
* curriculum helpers
* reusable robot kinematics
* generic tensor utilities
* logging builders
* domain-randomization utilities

Avoid passing the entire environment object when a narrow interface is sufficient.

Prefer:

```python
helper.update(error)
```

over:

```python
helper.update(env)
```

when possible.

---

# 42. Stateful Utilities

Stateful utilities must support parallel environments.

Their state should normally have shape:

```text
(num_envs, ...)
```

and live on the same device as the environment.

They should support partial reset:

```python
helper.reset(env_ids)
```

---

# 43. Logging Utilities

Complex logging logic may be extracted into `utils/`.

Logging helpers should:

* read tensors already calculated by the environment
* return diagnostic metrics
* not modify simulation state
* not modify reward
* not modify observation
* not alter policy behavior

Use:

```python
self.extras["log"] = ...
```

for training diagnostics.

---

# 44. Avoid Over-Abstraction

Do not move every small expression into `utils/`.

A helper belongs there when it is:

* reusable across tasks
* independently meaningful
* complex enough to obscure environment logic
* stateful enough to deserve encapsulation

Task-specific equations used once may remain inside the environment.

`utils/` must not become a miscellaneous dumping ground.

---

# 45. PPO Configuration

Every task must have:

```text
double_pendulum_rl/agents/<task>_ppo_cfg.py
```

with:

```python
@configclass
class <Task>PPORunnerCfg(
    RslRlOnPolicyRunnerCfg
):
    ...

    policy = RslRlPpoActorCriticCfg(
        ...
    )

    algorithm = RslRlPpoAlgorithmCfg(
        ...
    )
```

---

# 46. PPO Runner Configuration

Typical parameters include:

```text
num_steps_per_env
max_iterations
save_interval
experiment_name
obs_groups
```

Every task must use a unique experiment name.

With the fixed observation convention:

```python
return {
    "policy": observations,
}
```

the normal mapping is:

```python
obs_groups = {
    "policy": ["policy"],
    "critic": ["policy"],
}
```

---

# 47. Actor / Critic Configuration

The actor/critic config may vary between tasks.

Typical parameters include:

```text
initial exploration noise
observation normalization
actor hidden dimensions
critic hidden dimensions
activation
```

Network dimensions are task/training parameters, not fixed project-wide interface parameters.

---

# 48. PPO Algorithm Configuration

PPO parameters may vary between tasks.

Typical fields include:

```text
value_loss_coef
use_clipped_value_loss
clip_param
entropy_coef
num_learning_epochs
num_mini_batches
learning_rate
schedule
gamma
lam
desired_kl
max_grad_norm
```

Do not put environment logic into PPO configuration.

---

# 49. Gym Task Registration

Every task must be registered in:

```text
double_pendulum_rl/__init__.py
```

Registration connects:

```text
Gym task ID
    ↓
Environment
    ↓
Environment config
    ↓
PPO config
```

Use the project pattern:

```python
gym.register(
    id="WheelLeg-<Task>-Direct-v0",

    entry_point=(
        "double_pendulum_rl.env."
        "<task>_env:<Task>Env"
    ),

    disable_env_checker=True,

    kwargs={
        "env_cfg_entry_point": (
            "double_pendulum_rl.env."
            "<task>_env:<Task>EnvCfg"
        ),

        "rsl_rl_cfg_entry_point": (
            "double_pendulum_rl.agents."
            "<task>_ppo_cfg:"
            "<Task>PPORunnerCfg"
        ),
    },
)
```

A task is not integrated until it is registered.

Generic scripts must load tasks through the Gym registry.

---

# 50. Generic Infrastructure Contract

These scripts are generic infrastructure:

```text
scripts/smoke_test.py
scripts/train.py
scripts/play.py
```

They must not contain task-specific implementation.

A new task should only require changing the parameter section near the top of each script.

This project deliberately uses in-file parameter blocks for these scripts.
Do not replace them with `argparse`, CLI flags, environment variables, or a
second task-specific launcher unless the user explicitly requests that
interface. Runtime choices such as task ID, model path, environment count,
seed, headless mode, command, and checkpoint path should remain ordinary
constants in the top-level parameter section.

---

# 51. `scripts/smoke_test.py`

`smoke_test.py` must work with every registered environment.

Typical configurable parameters:

```python
TASK_NAME = "WheelLeg-<Task>-Direct-v0"

NUM_ENVS = ...
NUM_TEST_STEPS = ...
SEED = ...

ACTION_MODE = ...
ACTION_NOISE_SCALE = ...

HEADLESS = ...
```

The script should verify generic invariants:

* environment can be created
* reset succeeds
* action shape is `(num_envs, 6)`
* observation shape is `(num_envs, 28)`
* observations are finite
* rewards are finite
* stepping succeeds
* termination works
* timeout works
* automatic reset works
* no simulator exception occurs

The smoke test must not know task-specific reward semantics.

---

# 52. Forbidden Smoke-Test Pattern

Do not add task-name branches such as:

```python
if TASK_NAME == "WheelLeg-<SpecificTask>-Direct-v0":
    ...
```

to support ordinary tasks.

If the generic smoke test cannot test a new environment, first fix the environment interface.

---

# 53. `scripts/train.py`

`train.py` must train every registered task.

Only top-level configuration should change.

Typical parameters:

```python
TASK_NAME = "WheelLeg-<Task>-Direct-v0"
MODEL_NAME = ...
NUM_ENVS = ...
HEADLESS = ...

NUM_TRAINING_ITERATIONS = ...
RESUME_TRAINING = ...
RESUME_CHECKPOINT_PATH = ...
```

The script must load:

```text
environment configuration
PPO configuration
```

through the Gym registry.

It must not know:

* reward definitions
* command semantics
* joint semantics
* task-specific observation internals
* task-specific termination logic

---

# 54. Forbidden Training Pattern

This is normally an architecture violation:

```python
if TASK_NAME == "WheelLeg-<SpecificTask>-Direct-v0":
    ...
```

inside `train.py`.

First place task-specific behavior in:

```text
<Task>EnvCfg
<Task>Env
<Task>PPORunnerCfg
utils/
```

---

# 55. `scripts/play.py`

`play.py` must evaluate all registered policies through a common path.

Typical top-level parameters include:

```python
TASK_NAME = "WheelLeg-<Task>-Direct-v0"
MODEL_PATH = ...
NUM_ENVS = ...
HEADLESS = ...
FIXED_COMMAND = (..., ..., ...)
KEYBOARD_CONTROL = ...
```

Its generic responsibility is:

```text
launch simulator
        ↓
load registered task
        ↓
load policy
        ↓
policy inference
        ↓
environment step
        ↓
visualize / inspect
```

Task command generation remains inside the environment.

---

# 56. Forbidden Playback Pattern

Do not add:

```python
if TASK_NAME == "WheelLeg-<SpecificTask>-Direct-v0":
    # task-specific command logic
```

inside `play.py`.

If manual or interactive command control is needed later, implement a generic command-control interface rather than task-name branches.

---

# 57. Performance Rules

These methods run in the RL hot path:

```text
_sample_commands
_pre_physics_step
_apply_action
_get_observations
_get_dones
_get_rewards
```

Keep them vectorized.

Prefer:

```python
torch.sum(...)
torch.mean(...)
torch.where(...)
torch.clamp(...)
torch.stack(...)
torch.cat(...)
```

Avoid:

```text
Python loops over environments
.cpu()
.numpy()
unnecessary .item()
unnecessary CPU/GPU synchronization
```

Runtime tensors should normally use:

```python
device=self.device
```

and support arbitrary:

```python
self.num_envs
```

---

# 58. Numerical Safety

Environment code must protect both simulation and PPO from invalid numerical values.

Use when appropriate:

```python
torch.nan_to_num(...)
torch.isfinite(...)
torch.clamp(...)
```

Actions should be sanitized before reaching physics.

Observations must remain finite.

Rewards must remain finite.

Invalid environments must terminate/reset rather than propagating NaN or Inf.

---

# 59. New Task Workflow

When creating a new RL task, follow this sequence.

## Step 1: Start from the standard environment interface

Do not redesign:

```text
action dimension
action ordering
action semantics

observation dimension
observation ordering
observation semantics

command dimension
command ordering
command semantics
```

---

## Step 2: Create the environment

Create:

```text
double_pendulum_rl/env/<task>_env.py
```

Start from the standard environment structure.

---

## Step 3: Preserve the fixed-interface configuration

Copy and preserve the fixed-interface section of `EnvCfg`.

Do not modify normal policy-interface definitions.

---

## Step 4: Define task configuration

Set:

```text
episode length
command ranges
command resampling
reset randomization
termination thresholds
reward parameters
curriculum parameters
```

according to the task.

---

## Step 5: Reuse the standard lifecycle

Keep:

```text
__init__
_setup_scene
_sample_commands
_pre_physics_step
_apply_action
_get_observations
_get_dones
_reset_idx
```

as close as possible to the standard implementation.

Only make small task adaptations when required.

---

## Step 6: Design `_get_rewards()`

This is the primary task-specific implementation.

Define:

```text
desired behavior
tracking error
primary reward
necessary safety / regularization
```

Do not blindly copy another task's reward function.

---

## Step 7: Add utilities when justified

Move reusable or independently meaningful logic into:

```text
double_pendulum_rl/utils/
```

---

## Step 8: Create PPO config

Create:

```text
double_pendulum_rl/agents/<task>_ppo_cfg.py
```

---

## Step 9: Register the task

Edit:

```text
double_pendulum_rl/__init__.py
```

---

## Step 10: Run smoke test

Select the task in:

```text
scripts/smoke_test.py
```

Do not modify the core smoke-test implementation.

---

## Step 11: Train

Select the task in:

```text
scripts/train.py
```

Do not add task-specific branches.

---

## Step 12: Evaluate

Select the task/model in:

```text
scripts/play.py
```

Do not add task-specific branches.

---

# 60. MUST Rules

Coding agents MUST:

1. reuse `ROBOT_CFG`
2. keep robot definition separate from task logic
3. use `action_space = 6`
4. preserve the fixed action ordering
5. preserve the fixed action semantics
6. use `observation_space = 28`
7. preserve the fixed observation ordering
8. preserve the fixed observation meanings
9. use the fixed three-dimensional command interface
10. preserve command ordering and meaning
11. implement `_sample_commands(env_ids)`
12. keep commands per-environment
13. keep command sampling vectorized
14. preserve command/action/reward timing consistency
15. use explicit joint-name lookup
16. use `preserve_order=True` when order matters
17. support arbitrary `num_envs`
18. support subset reset through `env_ids`
19. keep hot-path code vectorized
20. put tunable values in `EnvCfg`
21. separate fixed-interface config from task config
22. treat `_get_rewards()` as the primary task-specific implementation
23. register every task in `double_pendulum_rl/__init__.py`
24. use the generic smoke test
25. use the generic training script
26. use the generic playback script

---

# 61. MUST NOT Rules

Coding agents MUST NOT:

1. redefine the robot for each task
2. redesign the action interface for an ordinary task
3. redesign the observation interface for an ordinary task
4. redesign the command interface for an ordinary task
5. remove unused command dimensions instead of fixing their ranges
6. change action ordering between tasks
7. change observation ordering between tasks
8. hard-code Isaac Lab joint indices
9. use per-environment Python loops in hot paths
10. allow old temporal state to leak through reset
11. resample a command before rewarding the action generated from the old command
12. scatter tunable magic numbers throughout methods
13. copy reward terms from another task without justification
14. put task reward logic in PPO configuration
15. put task logic in generic scripts
16. add ordinary task-name branches to `smoke_test.py`
17. add ordinary task-name branches to `train.py`
18. add ordinary task-name branches to `play.py`
19. create separate train/play scripts for ordinary tasks
20. modify stable lifecycle methods unnecessarily

---

# 62. Task-Specific Change Hierarchy

When adapting the environment for a new task, prefer changes in this order:

```text
1. reward configuration

2. _get_rewards()

3. command ranges / command curriculum

4. termination thresholds

5. reset randomization

6. small adaptations to command sampling

7. small adaptations to termination/reset logic

8. only then consider changing shared lifecycle behavior
```

Changing the fixed action/observation/command interface should be the last resort and should be treated as a project-wide redesign.

---

# 63. Environment Stability Principle

A well-structured collection of tasks should contain environment files whose structure remains highly similar.

The major differences should be visible mainly in:

```text
Task configuration
        +
_get_rewards()
```

If two ordinary task environments have completely different:

```text
action processing
observation construction
scene setup
reset pipeline
command lifecycle
```

assume the architecture has drifted and reconsider the implementation.

---

# 64. Completion Checklist

Before declaring a new task complete, verify:

## Environment

* [ ] `<task>_env.py` exists
* [ ] `<Task>EnvCfg` exists
* [ ] `<Task>Env` exists
* [ ] `ROBOT_CFG` is reused
* [ ] fixed action interface is preserved
* [ ] fixed observation interface is preserved
* [ ] fixed command interface is preserved
* [ ] action shape is 6
* [ ] observation shape is 28
* [ ] command shape is 3
* [ ] joint ordering is deterministic
* [ ] runtime tensors use `self.device`
* [ ] arbitrary `num_envs` is supported
* [ ] subset reset works
* [ ] hot paths are vectorized
* [ ] observations remain finite
* [ ] rewards remain finite

## Commands

* [ ] `self.commands` exists
* [ ] commands have shape `(num_envs, 3)`
* [ ] `_sample_commands(env_ids)` exists
* [ ] command ranges are defined in `EnvCfg`
* [ ] command sampling is vectorized
* [ ] command timers are per-environment
* [ ] command-dependent state resets correctly
* [ ] reset generates a valid initial command
* [ ] current action is rewarded against the correct command

## Rewards

* [ ] primary task objective is explicit
* [ ] reward terms are named
* [ ] tunable weights live in `EnvCfg`
* [ ] unnecessary shaping was not copied blindly
* [ ] continuous reward scaling uses `step_dt` where appropriate
* [ ] final reward is protected against NaN/Inf

## PPO

* [ ] `<task>_ppo_cfg.py` exists
* [ ] `<Task>PPORunnerCfg` exists
* [ ] experiment name is unique
* [ ] observation groups match environment output
* [ ] PPO config contains no environment logic

## Registration

* [ ] task is registered in `double_pendulum_rl/__init__.py`
* [ ] environment entry point is correct
* [ ] environment config entry point is correct
* [ ] PPO config entry point is correct

## Generic Infrastructure

* [ ] `scripts/smoke_test.py` required no task-specific implementation changes
* [ ] `scripts/train.py` required no task-specific implementation changes
* [ ] `scripts/play.py` required no task-specific implementation changes
* [ ] no task-name branches were added to generic scripts
* [ ] script runtime choices remain in the top-level in-file parameter blocks
* [ ] no CLI argument parser was introduced for ordinary script configuration

## Validation

* [ ] generic smoke test passes
* [ ] no NaN or Inf appears
* [ ] action shape is correct
* [ ] observation shape is correct
* [ ] automatic reset works
* [ ] training starts through generic `train.py`
* [ ] policy evaluation works through generic `play.py`

---

# 65. Final Environment Model

The intended environment architecture is:

```text
              PROJECT-WIDE FIXED INTERFACE

      action = 6
      observation = 28
      command = 3

      action ordering fixed
      observation ordering fixed
      command meaning fixed

                       │
                       ↓

              STANDARD ENV LIFECYCLE

      __init__               mostly fixed
      _setup_scene           mostly fixed
      _sample_commands       config-driven
      _pre_physics_step      almost fixed
      _apply_action          almost fixed
      _get_observations      almost fixed
      _get_dones             mostly fixed
      _reset_idx             mostly fixed

                       │
                       ↓

                 TASK DEFINITION

      command ranges
      command curriculum

      reward configuration

      _get_rewards()   ← primary task logic

      termination thresholds
      reset randomization

      small task-specific adaptations only when needed
```

---

# 66. Final Project Rule

For almost every future RL task, the workflow should be:

```text
create:
double_pendulum_rl/env/<task>_env.py

        ↓

preserve:
fixed action interface
fixed observation interface
fixed command interface
standard environment lifecycle

        ↓

modify:
task configuration

        ↓

design:
_get_rewards()

        ↓

create:
double_pendulum_rl/agents/<task>_ppo_cfg.py

        ↓

register:
double_pendulum_rl/__init__.py

        ↓

select task in:
scripts/smoke_test.py

        ↓

select task in:
scripts/train.py

        ↓

select task/model in:
scripts/play.py
```

The coding agent should think of a new task as:

> **A new reward objective and task configuration running on the same robot-policy interface.**

It should not treat a new task as an opportunity to redesign the environment architecture.
