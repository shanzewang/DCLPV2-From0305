# -*- coding: utf-8 -*-
"""
🔥 ORCA (Optimal Reciprocal Collision Avoidance) 控制器模块
=====================================================

完整的ORCA算法实现，用于多智能体避障导航

核心特性：
1. 基于Python-RVO2库的ORCA算法
2. 支持可配置的机器人避让比例
3. 多行人协同避障
4. 鲁棒的边界处理
5. 智能的目标更新机制

作者：基于HEIGHT项目的ORCA实现改进
版本：v2.0 - 完整生产级实现
"""

import numpy as np
from numpy.linalg import norm
import rvo2
import random
import time


class ORCAController:
    """
    🔥 ORCA控制器类 - 完整生产级实现

    该类管理单个行人的ORCA避障行为，包括：
    - ORCA速度计算
    - 与其他行人的协调
    - 机器人避让决策
    - 目标点管理
    - 卡住检测与恢复
    """

    def __init__(self, pedestrian_id, react_to_robot=True, config=None):
        """
        初始化ORCA控制器

        Args:
            pedestrian_id (int): 行人ID（robot_37-44对应0-7）
            react_to_robot (bool): 是否对机器人避让
            config (dict): 配置参数字典
        """
        self.pedestrian_id = pedestrian_id
        self.react_to_robot = react_to_robot  # 🔥 是否避让机器人

        # ============= 🔥 ORCA参数配置 =============
        if config is None:
            config = {}

        # ORCA核心参数（基于HEIGHT项目优化）
        self.neighbor_dist = config.get('neighbor_dist', 10.0)     # 邻居检测距离
        self.max_neighbors = config.get('max_neighbors', 10)       # 最大邻居数
        self.time_horizon = config.get('time_horizon', 5.0)        # 避障预测时间范围（秒）
        self.time_horizon_obst = config.get('time_horizon_obst', 5.0)  # 障碍物时间范围
        self.safety_space = config.get('safety_space', 0.15)       # 额外安全空间

        # 行人物理参数
        self.radius = config.get('pedestrian_radius', 0.3)         # 行人半径
        self.v_pref = config.get('v_pref', 0.4)                    # 偏好速度（基础）
        self.v_pref_range = config.get('v_pref_range', (0.3, 0.5))  # 速度随机范围
        self.max_speed = config.get('max_speed', 0.5)              # 最大速度

        # 机器人相关参数（更强的排斥）
        self.robot_radius = config.get('robot_radius', 0.15)        # 机器人半径
        self.robot_safety_factor = config.get('robot_safety_factor', 1.2)  # 机器人安全系数

        # ============= 🔥 运动状态变量 =============
        self.px = 0.0      # 当前位置x
        self.py = 0.0      # 当前位置y
        self.gx = 0.0      # 目标位置x
        self.gy = 0.0      # 目标位置y
        self.vx = 0.0      # 当前速度x
        self.vy = 0.0      # 当前速度y
        self.theta = 0.0   # 朝向角度

        # 速度在每次到达目标时随机化（模拟人类多样性）
        self.current_v_pref = np.random.uniform(*self.v_pref_range)

        # ============= 🔥 ORCA模拟器实例 =============
        self.sim = None           # RVO2模拟器实例
        self.agent_id = 0         # 在ORCA模拟器中的代理ID
        self.sim_initialized = False  # 模拟器是否已初始化

        # ============= 🔥 卡住检测与恢复机制 =============
        self.stuck_counter = 0            # 卡住计数器
        self.stuck_threshold = 3         # 卡住阈值（3个时间步）
        self.min_speed_threshold = 0.1    # 最小速度阈值（m/s）
        self.recent_speeds = []           # 最近速度历史
        self.recent_speed_window = 10     # 速度历史窗口大小

        # ============= 🔥 穿模检测机制（比卡住更严重）=============
        self.throughclipping_counter = 0        # 穿模计数器
        self.throughclipping_threshold = 10     # 穿模阈值（10个时间步）
        self.throughclipping_speed_threshold = 0.05  # 穿模速度阈值（几乎静止）

        # ============= 🔥 边界限制参数 =============
        self.area_center_x = 0.0
        self.area_half = 10
        self.boundary_margin = 0.5        # 边界预警距离

        # ============= 🔥 统计与调试 =============
        self.total_steps = 0              # 总步数
        self.goal_changes = 0             # 目标更换次数
        self.last_log_time = time.time()  # 上次日志时间

        print(f"🔥 ORCA控制器已创建: ID={pedestrian_id}, "
              f"避让机器人={'是' if react_to_robot else '否'}, "
              f"v_pref={self.current_v_pref:.2f}m/s")

    def initialize_position(self, px, py, gx, gy):
        """
        初始化行人位置和目标

        Args:
            px, py: 初始位置
            gx, gy: 目标位置
        """
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = 0.0
        self.vy = 0.0
        print(f"🚶 行人{self.pedestrian_id}: 初始化位置 ({px:.2f}, {py:.2f}) → 目标 ({gx:.2f}, {gy:.2f})")

    def _create_or_update_sim(self, other_pedestrians, robot_state=None):
        """
        🔥 创建或更新ORCA模拟器

        这是ORCA算法的核心，负责：
        1. 初始化RVO2模拟器
        2. 添加自己和其他代理
        3. 设置期望速度

        Args:
            other_pedestrians (list): 其他行人状态列表 [(px, py, vx, vy, radius), ...]
            robot_state (dict): 机器人状态 {'x': px, 'y': py, 'vx': vx, 'vy': vy}
        """
        # 计算需要添加的代理总数
        total_agents = 1 + len(other_pedestrians)  # 自己 + 其他行人
        if self.react_to_robot and robot_state is not None:
            total_agents += 1  # + 机器人

        # ============= 检查是否需要重新创建模拟器 =============
        need_recreate = False
        if self.sim is None:
            need_recreate = True
        elif self.sim.getNumAgents() != total_agents:
            need_recreate = True

        if need_recreate:
            # 🔥 删除旧的模拟器
            if self.sim is not None:
                del self.sim

            # 🔥 创建新的ORCA模拟器
            self.sim = rvo2.PyRVOSimulator(
                timeStep=0.1,                    # 时间步长
                neighborDist=self.neighbor_dist,
                maxNeighbors=self.max_neighbors,
                timeHorizon=self.time_horizon,
                timeHorizonObst=self.time_horizon_obst,
                radius=self.radius,              # 默认半径
                maxSpeed=self.max_speed          # 最大速度
            )

            # 🔥 添加自己（agent 0）
            self.agent_id = self.sim.addAgent(
                (self.px, self.py),                           # 当前位置
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.radius + self.safety_space,              # 有效半径
                self.current_v_pref,                          # 偏好速度
                (self.vx, self.vy)                           # 当前速度
            )

            # 🔥 添加其他行人
            for other_px, other_py, other_vx, other_vy, other_radius in other_pedestrians:
                self.sim.addAgent(
                    (other_px, other_py),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    other_radius + self.safety_space,
                    self.max_speed,
                    (other_vx, other_vy)
                )

            # 🔥 如果需要避让机器人，添加机器人
            if self.react_to_robot and robot_state is not None:
                robot_px = robot_state.get('x', 0.0)
                robot_py = robot_state.get('y', 0.0)
                robot_vx = robot_state.get('vx', 0.0)
                robot_vy = robot_state.get('vy', 0.0)

                # 机器人使用更大的有效半径（增强排斥效果）
                effective_robot_radius = self.robot_radius * self.robot_safety_factor + self.safety_space

                # print(f"--------------🔥 机器人{self.pedestrian_id}: factor={self.robot_safety_factor:.2f}m")

                self.sim.addAgent(
                    (robot_px, robot_py),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    effective_robot_radius,
                    self.max_speed,
                    (robot_vx, robot_vy)
                )

            self.sim_initialized = True
        else:
            # ============= 更新现有模拟器中的代理位置和速度 =============
            self.sim.setAgentPosition(self.agent_id, (self.px, self.py))
            self.sim.setAgentVelocity(self.agent_id, (self.vx, self.vy))

            # 更新其他行人
            agent_idx = 1
            for other_px, other_py, other_vx, other_vy, _ in other_pedestrians:
                self.sim.setAgentPosition(agent_idx, (other_px, other_py))
                self.sim.setAgentVelocity(agent_idx, (other_vx, other_vy))
                agent_idx += 1

            # 更新机器人（如果存在）
            if self.react_to_robot and robot_state is not None:
                robot_px = robot_state.get('x', 0.0)
                robot_py = robot_state.get('y', 0.0)
                robot_vx = robot_state.get('vx', 0.0)
                robot_vy = robot_state.get('vy', 0.0)
                self.sim.setAgentPosition(agent_idx, (robot_px, robot_py))
                self.sim.setAgentVelocity(agent_idx, (robot_vx, robot_vy))

        # ============= 设置期望速度 =============
        # 自己的期望：朝向目标
        dx = self.gx - self.px
        dy = self.gy - self.py
        dist_to_goal = np.sqrt(dx**2 + dy**2)

        if dist_to_goal > 0.01:
            # 归一化方向向量 * 偏好速度
            pref_vel_x = (dx / dist_to_goal) * self.current_v_pref
            pref_vel_y = (dy / dist_to_goal) * self.current_v_pref
        else:
            pref_vel_x = 0.0
            pref_vel_y = 0.0

        self.sim.setAgentPrefVelocity(self.agent_id, (pref_vel_x, pref_vel_y))

        # 🔥 其他行人：使用当前速度作为期望速度（表示他们想保持当前运动方向）
        agent_idx = 1
        for other_px, other_py, other_vx, other_vy, _ in other_pedestrians:
            # 使用当前速度作为期望速度的近似
            self.sim.setAgentPrefVelocity(agent_idx, (other_vx, other_vy))
            agent_idx += 1

        # 🔥 机器人（如果存在）：同样使用当前速度
        if self.react_to_robot and robot_state is not None:
            robot_vx = robot_state.get('vx', 0.0)
            robot_vy = robot_state.get('vy', 0.0)
            self.sim.setAgentPrefVelocity(agent_idx, (robot_vx, robot_vy))

    def compute_orca_velocity(self, other_pedestrians, robot_state=None):
        """
        🔥 计算ORCA最优避障速度

        这是对外的主要接口函数

        Args:
            other_pedestrians (list): 其他行人状态列表
            robot_state (dict): 机器人状态

        Returns:
            tuple: (vx_new, vy_new) 新的速度
        """
        # 🔥 创建或更新ORCA模拟器
        self._create_or_update_sim(other_pedestrians, robot_state)

        # 🔥 执行ORCA计算（一步）
        self.sim.doStep()

        # 🔥 获取计算出的最优速度
        vx_new, vy_new = self.sim.getAgentVelocity(self.agent_id)

        return vx_new, vy_new

   
    def update_state_from_ground_truth(self, real_px, real_py, vx, vy):
        """
        使用Stage返回的真实位置更新状态（解决穿墙问题）

        当启用cmd_vel物理驾驶时，Stage会处理碰撞。
        如果动态障碍物撞墙，Stage会阻止移动，实际位置不变。
        通过比较期望位置和实际位置，可以检测是否被墙阻挡。

        Args:
            real_px, real_py: Stage返回的真实位置
            vx, vy: 期望的速度
        """
        # 计算实际移动的速度（基于位置变化）
        dt = 0.1  # 控制周期
        actual_vx = (real_px - self.px) / dt
        actual_vy = (real_py - self.py) / dt

        # 使用真实位置更新
        self.px = real_px
        self.py = real_py
        self.vx = vx
        self.vy = vy

        # 边界硬限制
        self.px = max(self.area_center_x - self.area_half,
                     min(self.area_center_x + self.area_half, self.px))
        self.py = max(-self.area_half, min(self.area_half, self.py))

        # 更新朝向
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude > 0.1:
            self.theta = np.arctan2(vy, vx)

        # 记录实际速度历史（用于卡住检测）
        # 使用实际移动的速度，这样被墙挡住时速度会是0
        actual_v_magnitude = np.sqrt(actual_vx**2 + actual_vy**2)
        self.recent_speeds.append(actual_v_magnitude)
        if len(self.recent_speeds) > self.recent_speed_window:
            self.recent_speeds.pop(0)

        self.total_steps += 1

    def check_goal_reached(self, threshold=0.3):
        """
        检查是否到达目标

        Args:
            threshold: 到达阈值（米）

        Returns:
            bool: 是否到达
        """
        dist = norm([self.gx - self.px, self.gy - self.py])
        return dist < threshold

    def check_stuck(self):
        """
        🔥 检查是否卡住（基于HEIGHT论文的方法）

        判断标准：
        1. 最近3个时间步的平均速度 < 0.1 m/s
        2. 连续3个时间步速度过低

        Returns:
            bool: 是否卡住
        """
        if len(self.recent_speeds) < self.stuck_threshold:
            return False

        # 计算最近N步的平均速度
        recent_avg_speed = np.mean(self.recent_speeds[-self.stuck_threshold:])

        if recent_avg_speed < self.min_speed_threshold:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                return True
        else:
            self.stuck_counter = 0

        return False

    def check_throughclipping(self):
        """
        🔥 检查是否穿模（比卡住更严重的情况）

        穿模通常由于多个动态障碍物碰撞后重叠导致，
        此时障碍物无法移动，需要重新传送到新位置。

        判断标准：
        1. 最近10个时间步的平均速度 < 0.05 m/s（几乎静止）
        2. 连续10个时间步速度过低

        Returns:
            bool: 是否发生穿模
        """
        if len(self.recent_speeds) < self.throughclipping_threshold:
            return False

        # 计算最近10步的平均速度
        recent_avg_speed = np.mean(self.recent_speeds[-self.throughclipping_threshold:])

        if recent_avg_speed < self.throughclipping_speed_threshold:
            self.throughclipping_counter += 1
            if self.throughclipping_counter >= self.throughclipping_threshold:
                # 检测到穿模，重置计数器以便下次检测
                self.throughclipping_counter = 0
                return True
        else:
            self.throughclipping_counter = 0

        return False

    def set_new_goal(self, gx, gy):
        """
        设置新目标（到达旧目标或卡住时调用）

        Args:
            gx, gy: 新目标位置
        """
        self.gx = gx
        self.gy = gy
        self.goal_changes += 1

        # 🔥 每次换目标时，随机化速度（模拟人类多样性）
        if np.random.random() > 0.5:  # 50%概率调整速度
            self.current_v_pref = np.random.uniform(*self.v_pref_range)

        # 重置卡住检测
        self.stuck_counter = 0
        self.recent_speeds = []

        # print(f"🎯 行人{self.pedestrian_id}: 新目标 ({gx:.2f}, {gy:.2f}), "
        #       f"v_pref={self.current_v_pref:.2f}m/s")

    def reset_position_and_goal(self, px, py, gx, gy):
        """
        重置位置和目标（穿模恢复时调用）

        这个函数用于处理穿模情况，会完全重置障碍物的位置和目标。

        Args:
            px, py: 新的起始位置
            gx, gy: 新的目标位置
        """
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy

        # 重置速度为0
        self.vx = 0.0
        self.vy = 0.0

        # 随机化偏好速度
        self.current_v_pref = np.random.uniform(*self.v_pref_range)

        # 重置所有检测计数器和历史
        self.stuck_counter = 0
        self.throughclipping_counter = 0
        self.recent_speeds = []

        # 重置模拟器状态（下次compute_orca_velocity会重新创建）
        self.sim_initialized = False

        print(f"🔄 行人{self.pedestrian_id}: 穿模恢复 - 新位置({px:.2f}, {py:.2f}), 新目标({gx:.2f}, {gy:.2f})")

    def get_state(self):
        """
        获取当前状态（用于MPI通信）

        Returns:
            dict: 包含位置、速度、目标等信息的字典
        """
        return {
            'px': self.px,
            'py': self.py,
            'vx': self.vx,
            'vy': self.vy,
            'gx': self.gx,
            'gy': self.gy,
            'theta': self.theta,
            'v_pref': self.current_v_pref,
            'radius': self.radius,
            'react_to_robot': self.react_to_robot
        }

    def log_status(self, force=False):
        """
        打印状态日志（降低频率避免刷屏）

        Args:
            force: 是否强制打印
        """
        current_time = time.time()
        if force or (current_time - self.last_log_time > 5.0):  # 每5秒打印一次
            dist_to_goal = norm([self.gx - self.px, self.gy - self.py])
            v_magnitude = norm([self.vx, self.vy])

            # print(f"📊 行人{self.pedestrian_id}: "
            #       f"位置({self.px:.2f},{self.py:.2f}) → 目标({self.gx:.2f},{self.gy:.2f}) | "
            #       f"距离={dist_to_goal:.2f}m | 速度={v_magnitude:.2f}m/s | "
            #       f"避让机器人={'是' if self.react_to_robot else '否'} | "
            #       f"总步数={self.total_steps} | 换目标次数={self.goal_changes}")

            self.last_log_time = current_time


