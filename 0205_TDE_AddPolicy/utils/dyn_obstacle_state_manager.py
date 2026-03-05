# -*- coding: utf-8 -*-
"""
动态障碍物状态管理器
====================

本模块提供动态障碍物状态获取、转换和管理的统一接口。

核心功能：
    1. 获取机器人状态用于joint state构建
    2. 坐标系转换（世界坐标→机器人自我中心坐标）
    3. 动态障碍物状态提取和验证
    4. 支持可配置的障碍物数量（最多5个）

关键改进（相比原版）：
    - ✅ 移除时序维度（不再维护3帧历史）
    - ✅ 支持动态shape返回 [N, 6]
    - ✅ 支持最多5个动态障碍物（可配置）
    - ✅ 更强的数据有效性检查
    - ✅ 完整的中文注释
"""

import numpy as np
import rospy
import time
import math

# ============= 全局常量配置 =============

MAX_DYNAMIC_NUM = 5  # 最大支持的动态障碍物数量（默认5）
PEDESTRIAN_RADIUS = 0.3  # 行人半径（米）
MAX_DETECTION_DISTANCE = 5.0  # 最大检测距离（米），超过此距离的障碍物不纳入计算


class DynamicObstacleStateManager:
    """
    动态障碍物状态管理器类

    职责：
        - 获取机器人状态
        - 转换动态障碍物坐标（世界坐标→机器人自我中心坐标）
        - 验证障碍物数据有效性
        - 提供统一的障碍物状态获取接口

    属性：
        max_dynamic_num (int): 最大支持的动态障碍物数量

    使用示例：
        ```python
        # 初始化
        manager = DynamicObstacleStateManager(max_dynamic_num=5)

        # 获取当前动态障碍物
        obstacles = manager.get_current_dynamic_obstacles(env)
        # obstacles shape: [N, 6]，N为实际激活的障碍物数量
        ```
    """

    def __init__(self, max_dynamic_num=MAX_DYNAMIC_NUM):
        """
        初始化动态障碍物状态管理器

        参数：
            max_dynamic_num (int): 最大支持的动态障碍物数量，默认5
        """
        self.max_dynamic_num = max_dynamic_num
        rospy.loginfo(
            f"🔥 DynamicObstacleStateManager初始化: max_dynamic_num={max_dynamic_num}"
        )

    @staticmethod
    def get_robot_state_for_joint(env):
        """
        获取机器人状态用于构建joint state

        从环境中提取机器人的完整状态信息，用于后续的坐标转换和特征构建。

        参数：
            env: StageWorld环境实例

        返回：
            dict: 包含机器人完整状态信息的字典
                - 'px': float，机器人x坐标（米）
                - 'py': float，机器人y坐标（米）
                - 'vx': float，机器人x轴速度（米/秒）
                - 'vy': float，机器人y轴速度（米/秒）
                - 'theta': float，机器人朝向角度（弧度）
                - 'gx': float，目标点x坐标（米）
                - 'gy': float，目标点y坐标（米）
                - 'radius': float，机器人半径（米）
                - 'length1': float，机器人前向长度（米）
                - 'length2': float，机器人后向长度（米）
                - 'width': float，机器人宽度（米）

        异常处理：
            如果获取失败，返回默认零状态
        """
        try:
            # 获取机器人位姿 [x, y, theta]
            robot_pos = env.GetSelfStateGT()

            # 获取机器人速度 [v, w]（线速度和角速度）
            robot_speed = env.GetSelfSpeedGT()

            # 获取机器人尺寸 [length1, length2, width]
            robot_dims = env.get_robot_dimensions()

            # 计算笛卡尔坐标下的速度分量
            # vx = v * cos(theta)
            # vy = v * sin(theta)
            vx = robot_speed[0] * np.cos(robot_pos[2])
            vy = robot_speed[0] * np.sin(robot_pos[2])

            # 构建机器人状态字典
            robot_state = {
                'px': robot_pos[0],             # 机器人x坐标
                'py': robot_pos[1],             # 机器人y坐标
                'vx': vx,                       # 机器人x轴速度
                'vy': vy,                       # 机器人y轴速度
                'theta': robot_pos[2],          # 机器人朝向角度
                'gx': env.target_point[0],      # 目标点x坐标
                'gy': env.target_point[1],      # 目标点y坐标
                'radius': 0.3,                  # 机器人半径（可调整）
                'length1': robot_dims[0],       # 机器人前向长度
                'length2': robot_dims[1],       # 机器人后向长度
                'width': robot_dims[2]          # 机器人宽度
            }

            return robot_state

        except Exception as e:
            rospy.logwarn(f"❌ 获取机器人状态失败: {e}")
            # 返回默认状态
            return {
                'px': 0.0, 'py': 0.0, 'vx': 0.0, 'vy': 0.0, 'theta': 0.0,
                'gx': 0.0, 'gy': 0.0, 'radius': 0.3,
                'length1': 0.3, 'length2': 0.3, 'width': 0.2
            }

    @staticmethod
    def rotate_joint_state(robot_state, obstacle_state):
        """
        将障碍物特征转换为机器人自我中心坐标系

        此函数实现了从世界坐标系到机器人自我中心坐标系的转换。
        转换包括平移和旋转两个步骤：
        1. 平移：计算障碍物相对于机器人的位置
        2. 旋转：将相对位置旋转到机器人坐标系

        参数：
            robot_state (dict): 机器人状态信息
            obstacle_state (dict): 障碍物状态信息
                - 'px': float，障碍物x坐标（世界坐标系）
                - 'py': float，障碍物y坐标（世界坐标系）
                - 'vx': float，障碍物x轴速度（世界坐标系）
                - 'vy': float，障碍物y轴速度（世界坐标系）
                - 'radius': float，行人半径

        返回：
            np.array: 6维障碍物特征向量（机器人自我中心坐标系）
                [0] obstacle_px_local: 障碍物在局部坐标系x坐标
                [1] obstacle_py_local: 障碍物在局部坐标系y坐标
                [2] obstacle_vx_local: 障碍物在局部坐标系x轴速度
                [3] obstacle_vy_local: 障碍物在局部坐标系y轴速度
                [4] da: 机器人与障碍物当前距离（米）
                [5] r_ped: 行人半径（米）

        坐标转换公式：
            平移：相对位置 = 障碍物世界坐标 - 机器人世界坐标
            旋转：局部坐标 = 旋转矩阵 × 相对位置
                  其中旋转矩阵由机器人朝向θ决定

        距离限制：
            只有距离≤5米的障碍物才纳入计算，超过5米返回零向量
        """
        try:
            # ========== 1. 定义机器人自我中心坐标系 ==========
            # 旋转角度 = 机器人的全局朝向 (theta)
            robot_theta = robot_state['theta']
            cos_theta = np.cos(robot_theta)
            sin_theta = np.sin(robot_theta)

            # 辅助函数：将世界坐标系下的一个"点"转换到机器人坐标系
            def world_to_robot_frame_point(px_world, py_world, robot_px, robot_py, cos_t, sin_t):
                """
                坐标转换：世界坐标系 → 机器人坐标系（点）

                转换公式：
                    相对位置：
                        relative_x = px_world - robot_px
                        relative_y = py_world - robot_py

                    旋转到机器人坐标系：
                        local_x = relative_x * cos(θ) + relative_y * sin(θ)
                        local_y = -relative_x * sin(θ) + relative_y * cos(θ)
                """
                # 平移：计算相对位置
                relative_x = px_world - robot_px
                relative_y = py_world - robot_py

                # 旋转
                local_x = relative_x * cos_t + relative_y * sin_t
                local_y = -relative_x * sin_t + relative_y * cos_t
                return local_x, local_y

            # 辅助函数：将世界坐标系下的一个"向量"（如速度）转换到机器人坐标系（只旋转）
            def world_to_robot_frame_vector(vx_world, vy_world, cos_t, sin_t):
                """
                坐标转换：世界坐标系 → 机器人坐标系（向量）

                注意：向量转换只需旋转，不需要平移

                转换公式：
                    local_vx = vx_world * cos(θ) + vy_world * sin(θ)
                    local_vy = -vx_world * sin(θ) + vy_world * cos(θ)
                """
                local_vx = vx_world * cos_t + vy_world * sin_t
                local_vy = -vx_world * sin_t + vy_world * cos_t
                return local_vx, local_vy

            # ========== 2. 计算障碍物特征（Robot-Centric）==========
            # 转换障碍物全局位置到机器人局部坐标
            obstacle_px_local, obstacle_py_local = world_to_robot_frame_point(
                obstacle_state['px'], obstacle_state['py'],
                robot_state['px'], robot_state['py'],
                cos_theta, sin_theta
            )

            # 转换障碍物全局速度到机器人局部坐标
            obstacle_vx_local, obstacle_vy_local = world_to_robot_frame_vector(
                obstacle_state['vx'], obstacle_state['vy'],
                cos_theta, sin_theta
            )

            # 机器人与障碍物的欧几里得距离（在哪个坐标系计算都一样）
            da = np.sqrt((obstacle_state['px'] - robot_state['px'])**2 +
                        (obstacle_state['py'] - robot_state['py'])**2)

            #! ========== 3. 距离限制：只有5米以内的障碍物才纳入计算 ==========
            if da > MAX_DETECTION_DISTANCE:
                # 超过最大检测距离，返回零向量
                return np.zeros(6, dtype=np.float32)

            # 行人半径
            r_ped = obstacle_state['radius']

            # ========== 4. 构建6维障碍物特征 ==========
            obstacle_only_features = np.array([
                obstacle_px_local,   # 障碍物在局部坐标系x坐标
                obstacle_py_local,   # 障碍物在局部坐标系y坐标
                obstacle_vx_local,   # 障碍物在局部坐标系x轴速度
                obstacle_vy_local,   # 障碍物在局部坐标系y轴速度
                da,                  # 机器人与障碍物的距离
                r_ped                # 行人半径
            ], dtype=np.float32)

            return obstacle_only_features

        except Exception as e:
            rospy.logwarn(f"❌ Joint state坐标转换失败: {e}")
            # 返回6维零向量
            return np.zeros(6, dtype=np.float32)

    def get_current_dynamic_obstacles(self, env):
        """
        获取当前时刻的动态障碍物状态（无时序，只返回当前帧）

        这是核心方法，替代原来的Time_dynamic_obstacles函数。
        关键改进：
            - ✅ 移除时序维度（不再维护deque历史记录）
            - ✅ 返回动态shape [N, 6]，N为实际激活的障碍物数量
            - ✅ 支持最多max_dynamic_num个障碍物（默认5）
            - ✅ 增强的数据有效性检查

        参数：
            env: StageWorld环境实例

        返回：
            np.array: shape [N, 6]，N为当前激活的动态障碍物数量（0 ≤ N ≤ max_dynamic_num）
                     特征: [相对px, 相对py, 相对vx, 相对vy, 距离, 半径]
                     如果没有激活的障碍物，返回 [0, 6]

        数据流程：
            1. 实时MPI状态更新（如果是主控机器人）
            2. 获取机器人状态
            3. 遍历所有障碍物（1到max_dynamic_num）
            4. 验证数据有效性
            5. 坐标转换
            6. 返回 [N, 6] 数组

        使用示例：
            ```python
            manager = DynamicObstacleStateManager(max_dynamic_num=5)
            obstacles = manager.get_current_dynamic_obstacles(env)
            print(f"当前激活障碍物数量: {obstacles.shape[0]}")
            ```
        """
        try:
            # ========== 1. 实时MPI状态更新 ==========
            # 主控机器人需要先接收来自动态障碍物进程的MPI状态更新
            if env.is_main_robot:
                # 使用mpi_handler接收状态
                if hasattr(env, 'mpi_handler') and env.mpi_handler is not None:
                    for robot_id, position, velocity, active in env.mpi_handler.receive_dynamic_states():
                        # 更新环境的dynamic_obstacles_mpi_states
                        if robot_id in env.dynamic_obstacles_mpi_states:
                            env.dynamic_obstacles_mpi_states[robot_id].update({
                                'position': position,
                                'velocity': velocity,
                                'last_update': time.time(),
                                'active': active
                            })

            # ========== 2. 获取机器人状态 ==========
            robot_state = self.get_robot_state_for_joint(env)

            # ========== 3. 收集激活的障碍物 ==========
            joint_states = []

            # 遍历所有可能的动态障碍物（robot_1 到 robot_max_dynamic_num）
            for robot_id in range(1, self.max_dynamic_num + 1):
                # 检查MPI数据是否可用
                if (hasattr(env, 'dynamic_obstacles_mpi_states') and
                    robot_id in env.dynamic_obstacles_mpi_states):

                    obstacle_data = env.dynamic_obstacles_mpi_states[robot_id]

                    # ========== 数据有效性验证 ==========
                    # 验证1：检查是否激活
                    if not obstacle_data.get('active', False):
                        continue  # 跳过未激活的障碍物

                    # 验证2：检查时间戳是否新鲜（10秒内）
                    last_update = obstacle_data.get('last_update', 0)
                    if time.time() - last_update > 10.0:
                        continue  # 数据过期，跳过

                    # 验证3：检查是否有位置和速度数据
                    if 'position' not in obstacle_data or 'velocity' not in obstacle_data:
                        continue  # 缺少必要数据，跳过

                    # 验证4：检查数据长度
                    if (len(obstacle_data['position']) < 2 or
                        len(obstacle_data['velocity']) < 2):
                        continue  # 数据不完整，跳过

                    # ========== 提取障碍物信息 ==========
                    try:
                        obstacle_state = {
                            'px': float(obstacle_data['position'][0]),
                            'py': float(obstacle_data['position'][1]),
                            'vx': float(obstacle_data['velocity'][0]),
                            'vy': float(obstacle_data['velocity'][1]),
                            'radius': PEDESTRIAN_RADIUS  # 使用全局常量
                        }
                        
                        # print(f"!!!!!!!!!!!obstacle_state: {obstacle_state}")
                        # ========== 执行坐标转换 ==========
                        # 转换到机器人自我中心坐标系
                        joint_state = self.rotate_joint_state(robot_state, obstacle_state)

                        # 添加到列表（如果不是零向量）
                        if not np.allclose(joint_state, 0.0):
                            joint_states.append(joint_state)

                    except (ValueError, TypeError, IndexError) as e:
                        rospy.logwarn_throttle(10, f"⚠️ 解析障碍物{robot_id}数据失败: {e}")
                        continue

            # ========== 4. 返回结果 ==========
            if len(joint_states) > 0:
                # 有激活的障碍物，返回 [N, 6]
                result = np.array(joint_states, dtype=np.float32)
                rospy.logdebug(f"🔍 当前激活障碍物数量: {result.shape[0]}/{self.max_dynamic_num}")
                return result
            else:
                # 没有激活的障碍物，返回空数组 [0, 6]
                return np.zeros((0, 6), dtype=np.float32)

        except Exception as e:
            rospy.logwarn(f"❌ 获取动态障碍物时出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回安全的零矩阵 [0, 6]
            return np.zeros((0, 6), dtype=np.float32)

    def __repr__(self):
        """字符串表示"""
        return f"DynamicObstacleStateManager(max_dynamic_num={self.max_dynamic_num})"

