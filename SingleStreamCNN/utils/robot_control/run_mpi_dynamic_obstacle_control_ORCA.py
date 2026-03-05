# -*- coding: utf-8 -*-
"""
🔥 基于ORCA的MPI动态障碍物控制函数 - 完整替换版本
================================================================

这是完整的、生产级的ORCA控制实现，用于替换原有的社会力模型

核心改进：
1. ✅ 使用ORCA算法替代社会力模型
2. ✅ 支持可配置的机器人避让比例（20%-80%）
3. ✅ 多行人协同避障
4. ✅ 智能卡住检测与恢复
5. ✅ 鲁棒的边界处理
6. ✅ 完整的MPI通信支持

使用方法：
将此函数直接替换 torchdclp_1ped_MLP3*128_DRLVOrew_noback.py 中的
run_mpi_dynamic_obstacle_control 函数（143-483行）
"""

import rospy
import time
import random
import numpy as np
from geometry_msgs.msg import Pose2D, Twist
from utils.robot_control.orca_controller import ORCAController  # 导入我们创建的ORCA控制器


def get_safe_position_with_distance_check(env, existing_positions, min_distance=1.0, max_attempts=50):
    """
    生成一个与已有障碍物保持安全距离的随机位置

    Args:
        env: StageWorld环境实例
        existing_positions: 已有障碍物位置列表 [(px1, py1), (px2, py2), ...]
        min_distance: 最小间隔距离（米），默认1.0m
        max_attempts: 最大尝试次数

    Returns:
        tuple: (px, py) 安全位置坐标
    """
    for attempt in range(max_attempts):
        # 生成候选位置
        px, py = env._get_safe_random_position()

        # 检查与所有已有位置的距离
        is_safe = True
        for existing_px, existing_py in existing_positions:
            distance = np.sqrt((px - existing_px)**2 + (py - existing_py)**2)
            if distance < min_distance:
                is_safe = False
                break

        if is_safe:
            return px, py

    # 如果多次尝试都失败，返回最后一次生成的位置（降级策略）
    # rospy.logwarn(f"⚠️ 无法找到间隔{min_distance}m的安全位置，使用降级位置")
    return px, py


def run_mpi_dynamic_obstacle_control(robot_index, mpi_rank, mpi_comm):
    """
    🔥 基于ORCA的MPI动态障碍物控制函数 - 完整版

    功能：
    1. 使用ORCA算法计算避障速度
    2. 支持部分行人避让机器人
    3. 自动处理卡住和边界情况
    4. 完整的MPI状态同步

    Args:
        robot_index (int): 机器人索引 
        mpi_rank (int): MPI进程rank
        mpi_comm: MPI通信器

    特性：
        - 可配置避让比例（ROBOT_AVOID_RATIO）
        - 智能目标更新（到达或卡住）
        - 边界安全保护
        - 高效的ORCA计算
    """
    print(f"🔥 [ORCA版本] 启动MPI动态障碍物控制: robot_{robot_index} (rank {mpi_rank})")

    # ============= 🔥 ORCA配置参数 =============
    # 这些参数可以根据需要调整
    ORCA_CONFIG = {
        # ORCA核心参数（基于HEIGHT论文优化）
        'neighbor_dist': 10.0,           # 邻居检测距离（米）
        'max_neighbors': 10,             # 最大邻居数
        'time_horizon': 5.0,             # 避障预测时间（秒）- 重要！
        'time_horizon_obst': 5.0,        # 障碍物时间范围（秒）
        'safety_space': 0.23,            # 额外安全空间（米）

        # 行人物理参数
        'pedestrian_radius': 0.3,        # 行人半径（米）
        'v_pref': 0.25,                   # 基础偏好速度（米/秒）
        'v_pref_range': (0.1, 0.3),      # 速度随机范围（模拟人类多样性）
        'max_speed': 0.3,                # 最大速度（米/秒）

        # 机器人参数
        'robot_radius': 0.12,            # 机器人半径（米）
        'robot_safety_factor': 1.5,      # 机器人安全系数（增强排斥）
    }

    # 🔥 关键参数：机器人避让比例（可以动态调整）
    # 0.0 = 没有行人避让机器人
    # 0.2 = 20%行人避让机器人（推荐值，与HEIGHT论文一致）
    # 0.5 = 50%行人避让机器人
    # 1.0 = 所有行人避让机器人
    ROBOT_AVOID_RATIO = 0.2  # 🔥 可以修改这个值来调整避让比例

    # ============= 🔥 MPI标签定义 =============
    TAG_CURRICULUM_UPDATE = 100
    TAG_SHUTDOWN = 999
    TAG_ROBOT_STATE = 201            # 主控机器人状态
    TAG_PEDESTRIAN_STATES = 202      # 其他行人状态
    MAIN_ROBOT_RANK = 0

    try:
        # ============= 环境初始化 =============
        from stage_obs_dyn_curlearning_grid36_fixed_size import StageWorld

        # 创建环境连接（这里使用你项目中的StageWorldLogger）
        # 注意：由于原文件名包含特殊字符，这里使用StageWorld基类
        # 如果需要StageWorldLogger的特殊功能，请在主文件中修改
        env = StageWorld(540, index=robot_index, num_env=1)

        env.is_main_robot = False  # 标记为非主控机器人

        # 设置MPI通信参数
        env.set_mpi_comm(mpi_comm, mpi_rank, mpi_comm.Get_size())

        print(f"🔥 robot_{robot_index} 环境初始化完成")

        # 等待环境完全初始化
        time.sleep(3)

        # ============= 🔥 创建ORCA控制器 =============
        dynamic_obstacle_idx = robot_index - 1  # robot_1-8对应索引 0-7

        # 🔥 决定是否避让机器人（基于配置的比例）
        # 方法1：随机决定（每次运行不同）
        # react_to_robot = (random.random() < ROBOT_AVOID_RATIO)

        # 方法2：固定分配（可预测）- 可选
        # total_pedestrians = 8
        # num_avoid = int(total_pedestrians * ROBOT_AVOID_RATIO)
        # react_to_robot = (dynamic_obstacle_idx < num_avoid)

        #方法3∶精确指定哪些行人避让机器人 ★当前使用
        # #只让 robot 1
        # #方法3∶精确指定哪些行人避让机器人 
        AVOID_LIST =[] #robot 1对应索引0，robot 2对应索引1，以此类推
        react_to_robot = (dynamic_obstacle_idx in AVOID_LIST)



        orca_controller = ORCAController(
            pedestrian_id=dynamic_obstacle_idx,
            react_to_robot=react_to_robot,
            config=ORCA_CONFIG
        )

        print(f"🔥 robot_{robot_index} ORCA控制器已创建: "
              f"索引={dynamic_obstacle_idx}, "
              f"避让机器人={'是' if react_to_robot else '否'}")

        # ============= 🔥 检查初始课程学习状态 =============
        # print(f"🔥 robot_{robot_index} 检查初始课程学习状态...")
        # env.check_initial_curriculum_state()

        # ============= 初始化行人位置和目标（Plan A：启动时直接激活） =============
        # Plan A：启动时全部激活8个障碍物，不依赖课程学习消息
        is_activated = True
        rospy.loginfo(f"robot_{robot_index} 启动时直接激活（Plan A）")

        last_curriculum_check = time.time()

        # ============= 🔥 生成初始位置（带距离检测）=============
        print(f"🔍 robot_{robot_index} 正在生成安全初始位置...")

        # 🔥 关键：从MPI获取其他已生成障碍物的位置
        # 等待一段时间让其他进程先生成位置
        time.sleep(0.2 * dynamic_obstacle_idx)  # 每个障碍物错开0.2秒

        # 尝试从MPI接收其他障碍物的位置
        existing_positions = []
        if hasattr(env, 'dynamic_obstacles_mpi_states'):
            for other_robot_id, other_state in env.dynamic_obstacles_mpi_states.items():
                if other_robot_id != robot_index and other_state.get('active', False):
                    pos = other_state.get('position', [0, 0, 0])
                    existing_positions.append((pos[0], pos[1]))

        # 生成与其他障碍物保持1m以上距离的起点
        px, py = get_safe_position_with_distance_check(env, existing_positions, min_distance=1.0)
        gx, gy = env._get_safe_random_position()
        orca_controller.initialize_position(px, py, gx, gy)
        rospy.loginfo(f"robot_{robot_index} ORCA初始化完成: 起点({px:.2f}, {py:.2f}), 目标({gx:.2f}, {gy:.2f})")

        # 地图外位置（未激活时使用）
        # outside_position = env._get_fixed_outside_position(robot_index)

        # 创建ROS发布器
        pose_publisher = env.pose_publisher
        cmd_publisher = env.cmd_vel

        # 🔥 关键修复：发布初始位置到Stage（只在初始化时使用pose传送）
        # 这样机器人会从随机位置开始，而不是world文件中的初始位置
        initial_pose = Pose2D()
        initial_pose.x = px
        initial_pose.y = py
        initial_pose.theta = 0.0
        pose_publisher.publish(initial_pose)
        rospy.sleep(0.1)  # 等待Stage更新位置
        print(f"✅ robot_{robot_index} 初始位置已发布到Stage: ({px:.2f}, {py:.2f})")

        print(f"🔥 robot_{robot_index} 发布器确认: "
              f"pose={pose_publisher is not None}, "
              f"cmd={cmd_publisher is not None}")

        # ============= 🔥 主控制循环 =============
        rate = rospy.Rate(10)  # 10Hz控制频率
        loop_counter = 0

        # 🔥 关键修复：在循环外初始化变量
        robot_state = None
        other_pedestrians_states = []

        print(f"🚀 robot_{robot_index} 开始ORCA控制循环")

        while not rospy.is_shutdown():
            try:
                loop_counter += 1

                # ============= 🔥 接收所有行人状态字典（Tag 202）=============
                # 主控机器人通过Tag 202广播所有动态障碍物的状态字典
                if mpi_comm and mpi_comm.Iprobe(source=MAIN_ROBOT_RANK, tag=TAG_PEDESTRIAN_STATES):
                    try:
                        # 接收包含所有行人状态的字典
                        all_pedestrians_dict = mpi_comm.recv(source=MAIN_ROBOT_RANK, tag=TAG_PEDESTRIAN_STATES)

                        # 更新环境的all_dynamic_states属性
                        if all_pedestrians_dict and isinstance(all_pedestrians_dict, dict):
                            env.all_dynamic_states = all_pedestrians_dict

                            # 调试信息（可选，降低打印频率）
                            if loop_counter % 100 == 0:
                                print(f"🔄 robot_{robot_index} 收到Tag 202状态更新，包含{len(all_pedestrians_dict)}个行人")
                    except Exception as e:
                        print(f"❌ robot_{robot_index} 处理Tag 202消息失败: {e}")


                # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!env.all_dynamic_states: {env.all_dynamic_states}")
                # ============= 🔥 每个循环都更新其他行人状态（修复bug）=============
                # 🔥 提取其他行人状态（用于ORCA协同避障）
                other_pedestrians_states = []
                if hasattr(env, 'all_dynamic_states') and env.all_dynamic_states:
                    for other_robot_id, other_state in env.all_dynamic_states.items():
                        if other_robot_id == robot_index:
                            continue  # 跳过自己

                        # ========== 数据有效性检查 ==========
                        # 检查1：是否处于激活状态
                        if not other_state.get('active', False):
                            continue  # 跳过未激活的行人

                        # 检查2：时间戳是否新鲜（5秒内）
                        last_update = other_state.get('last_update', 0)
                        if time.time() - last_update > 5.0:
                            continue  # 数据过期，跳过

                        # 检查3：是否有有效的位置和速度数据
                        if 'position' not in other_state or 'velocity' not in other_state:
                            continue  # 缺少必要数据，跳过

                        # ========== 提取行人信息（ORCA需要的格式）==========
                        try:
                            # 正确提取position和velocity（它们是列表或元组）
                            position = other_state['position']
                            velocity = other_state['velocity']

                            other_px = float(position[0])
                            other_py = float(position[1])
                            other_vx = float(velocity[0])
                            other_vy = float(velocity[1])
                            other_radius = ORCA_CONFIG['pedestrian_radius']

                            # 添加到其他行人列表
                            other_pedestrians_states.append(
                                (other_px, other_py, other_vx, other_vy, other_radius)
                            )

                        except (IndexError, TypeError, ValueError) as e:
                            print(f"⚠️ robot_{robot_index} 解析行人{other_robot_id}数据失败: {e}")
                            continue

                # print(f"-------------------------other_pedestrians_states={other_pedestrians_states}")

                # 🔥 接收主控机器人状态（用于ORCA计算）
                if mpi_comm and mpi_comm.Iprobe(source=MAIN_ROBOT_RANK, tag=TAG_ROBOT_STATE):
                    robot_state = mpi_comm.recv(source=MAIN_ROBOT_RANK, tag=TAG_ROBOT_STATE)
                
                # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!robot_state={robot_state}")

                # ============= 🔥 定期检查课程学习状态和MPI消息 =============
                current_time = time.time()
                if current_time - last_curriculum_check >= 2.0:

                    # 🔥 接收课程学习更新
                    if mpi_comm and mpi_comm.Iprobe(source=MAIN_ROBOT_RANK, tag=TAG_CURRICULUM_UPDATE):
                        message = mpi_comm.recv(source=MAIN_ROBOT_RANK, tag=TAG_CURRICULUM_UPDATE)

                        if message.get("command") == "curriculum_update":
                            active_dynamic_count = message.get("active_dynamic", 0)
                            new_is_activated = dynamic_obstacle_idx < active_dynamic_count

                            if new_is_activated != is_activated:
                                is_activated = new_is_activated
                                status = "激活" if is_activated else "隐藏"
                                print(f"🔄 robot_{robot_index} 状态变更: {status}")

                                # 激活时重新初始化位置
                                if is_activated:
                                    px, py = env._get_safe_random_position()
                                    gx, gy = env._get_safe_random_position()
                                    orca_controller.initialize_position(px, py, gx, gy)

                                    # 发布新位置到Stage
                                    reactivate_pose = Pose2D()
                                    reactivate_pose.x = px
                                    reactivate_pose.y = py
                                    reactivate_pose.theta = 0.0
                                    pose_publisher.publish(reactivate_pose)
                                    print(f"🔄 robot_{robot_index} 重新激活，位置重置到 ({px:.2f}, {py:.2f})")

                    # 🔥 检查结束信号
                    if mpi_comm and mpi_comm.Iprobe(source=MAIN_ROBOT_RANK, tag=TAG_SHUTDOWN):
                        message = mpi_comm.recv(source=MAIN_ROBOT_RANK, tag=TAG_SHUTDOWN)
                        print(f"🛑 robot_{robot_index}: 收到结束信号")
                        break

                    last_curriculum_check = current_time

                # ============= 🔥 位置和速度控制 =============
                cmd = Twist()

                if is_activated:
                    # ================================================
                    # 🔥 ORCA算法控制逻辑（核心部分）
                    # 修改：使用cmd_vel物理驾驶 + Stage真实位置反馈
                    # ================================================

                    # 1️⃣ 使用ORCA计算最优避障速度
                    try:
                        vx_new, vy_new = orca_controller.compute_orca_velocity(
                            other_pedestrians=other_pedestrians_states,
                            robot_state=robot_state if orca_controller.react_to_robot else None
                        )
                    except Exception as e:
                        print(f"❌ robot_{robot_index} ORCA计算失败: {e}")
                        # 降级策略：直线朝向目标
                        state = orca_controller.get_state()
                        dx = state['gx'] - state['px']
                        dy = state['gy'] - state['py']
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist > 0.01:
                            vx_new = (dx / dist) * state['v_pref']
                            vy_new = (dy / dist) * state['v_pref']
                        else:
                            vx_new, vy_new = 0.0, 0.0

                    # 2️⃣ 设置并发布速度指令（物理驾驶）
                    cmd.linear.x = vx_new
                    cmd.linear.y = vy_new
                    cmd.angular.z = 0.0
                    cmd_publisher.publish(cmd)  # 启用物理驾驶

                    # 3️⃣ 从Stage获取真实位置，更新ORCA状态
                    # Stage物理引擎会处理碰撞，如果撞墙则实际位置不变
                    real_px, real_py, real_theta = env.GetSelfStateGT()
                    orca_controller.update_state_from_ground_truth(real_px, real_py, vx_new, vy_new)

                    # 4️⃣ 检查是否到达目标
                    if orca_controller.check_goal_reached(threshold=0.3):
                        gx, gy = env._get_safe_random_position()
                        orca_controller.set_new_goal(gx, gy)
                        # print(f"✅ robot_{robot_index} 到达目标，生成新目标")

                    # 5️⃣ 检查是否卡住（现在基于真实位置变化检测，碰墙时速度为0）
                    if orca_controller.check_stuck():
                        gx, gy = env._get_safe_random_position()
                        orca_controller.set_new_goal(gx, gy)
                        # if loop_counter % 50 == 0:  # 降低打印频率
                        #     print(f"⚠️ robot_{robot_index} 检测到卡住（可能碰到墙），重新设置目标: ({gx:.2f}, {gy:.2f})")

                    # 🔥 6️⃣ 检查是否穿模（比卡住更严重，需要重置位置）
                    if orca_controller.check_throughclipping():
                        print(f"❌ robot_{robot_index} 检测到穿模，重置位置和目标")

                        # 获取其他障碍物位置，避免重置到相同位置
                        existing_positions = []
                        if hasattr(env, 'all_dynamic_states'):
                            for other_robot_id, other_state in env.all_dynamic_states.items():
                                if other_robot_id != robot_index and other_state.get('active', False):
                                    pos = other_state.get('position', [0, 0, 0])
                                    existing_positions.append((pos[0], pos[1]))

                        # 生成新的起点和目标，确保间隔1m以上
                        new_px, new_py = get_safe_position_with_distance_check(
                            env, existing_positions, min_distance=1.0
                        )
                        new_gx, new_gy = env._get_safe_random_position()

                        # 重置ORCA控制器的位置和目标
                        orca_controller.reset_position_and_goal(new_px, new_py, new_gx, new_gy)

                        # 发布新位置到Stage（传送）
                        teleport_pose = Pose2D()
                        teleport_pose.x = new_px
                        teleport_pose.y = new_py
                        teleport_pose.theta = 0.0
                        pose_publisher.publish(teleport_pose)
                        rospy.sleep(0.1)

                        print(f"✅ robot_{robot_index} 穿模恢复完成: 新位置({new_px:.2f}, {new_py:.2f})")

                    # 7️⃣ 获取更新后的状态用于MPI发送
                    state = orca_controller.get_state()
                    px = state['px']
                    py = state['py']
                    vx = state['vx']
                    vy = state['vy']
                    theta = state['theta']

                else:
                    # 未激活状态：速度置零
                    cmd.linear.x = 0.0
                    cmd.linear.y = 0.0
                    cmd.angular.z = 0.0
                    cmd_publisher.publish(cmd)

                    # 获取当前位置用于MPI
                    real_px, real_py, real_theta = env.GetSelfStateGT()
                    px, py = real_px, real_py
                    vx, vy = 0.0, 0.0

                # ============= 🔥 更新MPI状态（不再使用pose传送）=============
                # 注意：不再使用pose_publisher传送位置，改用cmd_vel物理驾驶
                # pose_publisher.publish(pose)  # 已弃用

                # 🔥 更新env.object_state，以便MPI状态发送
                env.object_state = [px, py, vx, vy]

                # 🔥 手动调用MPI状态发送
                env.send_my_state_to_main_process()

                rate.sleep()

            except rospy.ROSInterruptException:
                break
            except Exception as e:
                print(f"❌ robot_{robot_index} 控制循环错误: {e}")
                import traceback
                traceback.print_exc()
                rate.sleep()

    except Exception as e:
        print(f"❌ robot_{robot_index} 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"✅ robot_{robot_index} ORCA控制结束")

