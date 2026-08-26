from isaaclab.app import AppLauncher

# 启动sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from double_pendulum_rl.robot import ROBOT_CFG

import carb
import omni.appwindow


# 定义场景配置类
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # 灯光
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    # 机器人
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # 创建仿真上下文
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,  # 物理仿真频率120Hz
    )
    sim = SimulationContext(sim_cfg)

    # 设置初始相机
    sim.set_camera_view(
        eye=(2.0, 2.0, 1.5),
        target=(0.0, 0.0, 0.4),
    )

    # 创建仿真环境
    scene_cfg = RobotSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )
    scene = InteractiveScene(scene_cfg)

    ## 实现按R重启
    # 获取 robot
    robot = scene["robot"]
    reset_requested = [False]
    # 定义键盘回调
    def on_keyboard_event(event, *args):
        if (
            event.type == carb.input.KeyboardEventType.KEY_PRESS
            and event.input == carb.input.KeyboardInput.R
        ):
            reset_requested[0] = True
        return True
    # 注册键盘回调
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_subscription = input_interface.subscribe_to_keyboard_events(
        keyboard,
        on_keyboard_event,
    )
    # 定义重启函数
    def reset_robot():
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins

        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.reset()

        print("[INFO] Robot reset.")

    # 初始化仿真
    sim.reset()
    reset_robot()

    # 打印关节顺序
    print("[INFO] Joint order:")
    for index, name in enumerate(robot.joint_names):
        print(f"  {index}: {name}")

    # 仿真循环
    sim_dt = sim.get_physics_dt()
    step_count = 0
    while simulation_app.is_running():
        # 是否重启
        if reset_requested[0]:
            reset_robot()
            reset_requested[0] = False
        # 计算仿真数据
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        # 打印各种状态
        step_count += 1
        if step_count % 120 == 0:
            joint_pos = robot.data.joint_pos[0]
            joint_vel = robot.data.joint_vel[0]

            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0]
            root_lin_vel = robot.data.root_lin_vel_w[0]
            root_ang_vel = robot.data.root_ang_vel_w[0]

            print(f"\n[STATE] step={step_count}")
            print("  joint_pos:", joint_pos.detach().cpu().tolist())
            print("  joint_vel:", joint_vel.detach().cpu().tolist())
            print("  root_pos:", root_pos.detach().cpu().tolist())
            print("  root_quat:", root_quat.detach().cpu().tolist())
            print("  root_lin_vel:", root_lin_vel.detach().cpu().tolist())
            print("  root_ang_vel:", root_ang_vel.detach().cpu().tolist())

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
