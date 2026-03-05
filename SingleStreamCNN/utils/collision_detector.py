# -*- coding: utf-8 -*-
"""
碰撞检测模块
============

功能：为机器人导航系统提供高效、准确的碰撞检测功能

主要特性：
    1. 支持矩形包围盒碰撞检测
    2. 支持圆形碰撞检测
    3. 支持动态障碍物状态过滤
    4. 可配置的安全边界
    5. 详细的碰撞信息返回

作者：Claude Code
创建时间：2025-12-23
版本：1.0.0
"""

import math
import time
import numpy as np
import rospy


class CollisionDetector:
    """
    碰撞检测器类

    提供多种碰撞检测方法，支持机器人与动态障碍物的碰撞判断
    """

    def __init__(self,
                 robot_length1=0.3,      # 机器人前半长
                 robot_length2=0.3,      # 机器人后半长
                 robot_width=0.3,        # 机器人宽度
                 pedestrian_radius=0.3,  # 行人半径
                 collision_mode='rectangle',  # 碰撞检测模式：'rectangle' 或 'circle'
                 safety_margin=0.0,      # 额外的安全边界
                 state_timeout=5.0):     # 状态超时时间（秒）
        """
        初始化碰撞检测器

        Args:
            robot_length1: 机器人前半部长度（米）
            robot_length2: 机器人后半部长度（米）
            robot_width: 机器人宽度（米）
            pedestrian_radius: 行人/障碍物半径（米）
            collision_mode: 碰撞检测模式
                - 'rectangle': 使用矩形包围盒检测（更精确）
                - 'circle': 使用圆形检测（计算更快）
            safety_margin: 额外的安全边界（米），增加碰撞检测灵敏度
            state_timeout: 障碍物状态超时时间（秒），过期状态将被忽略
        """
        self.robot_length1 = robot_length1
        self.robot_length2 = robot_length2
        self.robot_width = robot_width
        self.pedestrian_radius = pedestrian_radius
        self.collision_mode = collision_mode
        self.safety_margin = safety_margin
        self.state_timeout = state_timeout

        # 预计算机器人有效半径（用于圆形检测模式）
        self.robot_effective_radius = math.sqrt(
            (robot_width)**2 + (robot_length1 + robot_length2)**2
        ) / 2

        # rospy.loginfo(f"CollisionDetector 初始化完成: mode={collision_mode}, "
        #              f"robot_size=({robot_length1:.2f}, {robot_length2:.2f}, {robot_width:.2f}), "
        #              f"ped_radius={pedestrian_radius:.2f}")

    def check_pedestrian_collision(self,
                                   robot_position,      # (x, y)
                                   robot_orientation,   # (cos_theta, sin_theta)
                                   obstacles_states,    # dict: {robot_id: state}
                                   check_ids=None):     # list: 要检查的障碍物ID列表
        """
        检测机器人与动态障碍物的碰撞

        Args:
            robot_position: 机器人位置元组 (x, y)
            robot_orientation: 机器人朝向元组 (cos_theta, sin_theta)
            obstacles_states: 动态障碍物状态字典
                格式：{robot_id: {'position': [px, py, pz], 'active': bool, 'last_update': timestamp}}
            check_ids: 要检查的障碍物ID列表，None表示检查所有

        Returns:
            collision_info: 字典，包含碰撞检测结果
                {
                    'detected': bool,           # 是否检测到碰撞
                    'collided_ids': list,       # 发生碰撞的障碍物ID列表
                    'closest_distance': float,  # 最近障碍物的距离
                    'details': list             # 每个障碍物的详细信息
                }
        """
        x, y = robot_position
        cos_theta, sin_theta = robot_orientation

        # 初始化返回信息
        collision_info = {
            'detected': False,
            'collided_ids': [],
            'closest_distance': float('inf'),
            'details': []
        }

        # 确定要检查的障碍物ID
        if check_ids is None:
            check_ids = list(obstacles_states.keys())

        current_time = time.time()

        # 遍历所有需要检查的障碍物
        for robot_id in check_ids:
            if robot_id not in obstacles_states:
                continue

            obstacle_state = obstacles_states[robot_id]

            # 过滤掉不活动的障碍物
            if not obstacle_state.get('active', False):
                continue

            # 过滤掉状态过期的障碍物
            last_update = obstacle_state.get('last_update', 0)
            if current_time - last_update >= self.state_timeout:
                continue

            # 获取障碍物位置
            position = obstacle_state.get('position', [0.0, 0.0, 0.0])
            px_i, py_i = position[0], position[1]

            # 过滤掉无效位置（0,0）
            if px_i == 0.0 and py_i == 0.0:
                continue

            # 计算相对位置
            dx = px_i - x
            dy = py_i - y
            dist_center_to_center = math.sqrt(dx**2 + dy**2)

            # 记录最近距离
            if dist_center_to_center < collision_info['closest_distance']:
                collision_info['closest_distance'] = dist_center_to_center

            # 根据检测模式判断碰撞
            is_collision = False

            if self.collision_mode == 'rectangle':
                # 矩形包围盒检测（更精确）
                is_collision = self._check_rectangle_collision(
                    dx, dy, cos_theta, sin_theta
                )
            elif self.collision_mode == 'circle':
                # 圆形检测（更快）
                is_collision = self._check_circle_collision(dist_center_to_center)

            # 记录碰撞信息
            obstacle_detail = {
                'robot_id': robot_id,
                'position': (px_i, py_i),
                'distance': dist_center_to_center,
                'collision': is_collision
            }
            collision_info['details'].append(obstacle_detail)

            if is_collision:
                collision_info['detected'] = True
                collision_info['collided_ids'].append(robot_id)
                rospy.logwarn_throttle(1, f"检测到与障碍物 {robot_id} 的碰撞! 距离={dist_center_to_center:.2f}m")

        return collision_info

    def _check_rectangle_collision(self, dx, dy, cos_theta, sin_theta):
        """
        使用矩形包围盒检测碰撞（机器人局部坐标系）

        Args:
            dx, dy: 障碍物相对于机器人的世界坐标偏移
            cos_theta, sin_theta: 机器人朝向的cos和sin值

        Returns:
            bool: 是否发生碰撞
        """
        # 将障碍物位置转换到机器人局部坐标系
        # 机器人坐标系：x轴指向前方，y轴指向左侧
        robot_x = dx * cos_theta + dy * sin_theta
        robot_y = -dx * sin_theta + dy * cos_theta

        # 计算碰撞边界（考虑行人半径和安全边界）
        ped_effective_radius = self.pedestrian_radius + self.safety_margin

        front_boundary = self.robot_length1 + ped_effective_radius
        back_boundary = -(self.robot_length2 + ped_effective_radius)
        side_boundary = self.robot_width + ped_effective_radius

        # 判断障碍物是否在矩形包围盒内
        is_within_front_back = (back_boundary <= robot_x <= front_boundary)
        is_within_left_right = (abs(robot_y) <= side_boundary)

        return is_within_front_back and is_within_left_right

    def _check_circle_collision(self, distance):
        """
        使用圆形检测碰撞（中心到中心距离）

        Args:
            distance: 机器人中心到障碍物中心的距离

        Returns:
            bool: 是否发生碰撞
        """
        collision_threshold = (
            self.robot_effective_radius +
            self.pedestrian_radius +
            self.safety_margin
        )

        return distance <= collision_threshold

    def update_robot_size(self, length1, length2, width):
        """
        更新机器人尺寸参数

        Args:
            length1: 前半长
            length2: 后半长
            width: 宽度
        """
        self.robot_length1 = length1
        self.robot_length2 = length2
        self.robot_width = width

        # 重新计算有效半径
        self.robot_effective_radius = math.sqrt(
            (width)**2 + (length1 + length2)**2
        ) / 2

        # rospy.loginfo(f"CollisionDetector 机器人尺寸已更新: "
        #              f"({length1:.2f}, {length2:.2f}, {width:.2f})")

    def update_pedestrian_radius(self, radius):
        """
        更新行人半径参数

        Args:
            radius: 行人半径（米）
        """
        self.pedestrian_radius = radius
        # rospy.loginfo(f"CollisionDetector 行人半径已更新: {radius:.2f}m")

    def set_collision_mode(self, mode):
        """
        设置碰撞检测模式

        Args:
            mode: 'rectangle' 或 'circle'
        """
        if mode not in ['rectangle', 'circle']:
            rospy.logwarn(f"无效的碰撞检测模式: {mode}，保持当前模式")
            return

        self.collision_mode = mode
        # rospy.loginfo(f"CollisionDetector 碰撞检测模式已设置为: {mode}")

    def get_collision_statistics(self, collision_info):
        """
        获取碰撞统计信息

        Args:
            collision_info: check_pedestrian_collision()返回的结果

        Returns:
            dict: 统计信息
                {
                    'total_checked': int,      # 检查的障碍物总数
                    'collision_count': int,    # 碰撞数量
                    'closest_distance': float, # 最近距离
                    'collision_rate': float    # 碰撞率
                }
        """
        total_checked = len(collision_info['details'])
        collision_count = len(collision_info['collided_ids'])

        return {
            'total_checked': total_checked,
            'collision_count': collision_count,
            'closest_distance': collision_info['closest_distance'],
            'collision_rate': collision_count / max(total_checked, 1)
        }
