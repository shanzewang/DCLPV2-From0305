# -*- coding: utf-8 -*-
"""
MPI通信管理器
=============

本模块提供MPI多进程通信的统一管理接口，封装所有MPI相关的通信逻辑。

核心功能：
    1. MPI初始化和配置管理
    2. 进程角色识别（主控/目标发布/动态障碍物）
    3. 状态发送/接收的统一接口
    4. 课程学习状态广播
    5. 机器人和行人状态同步

设计原则：
    - 单一职责：只负责MPI通信
    - 接口清晰：提供简洁的API
    - 错误处理：完善的异常捕获
    - 高性能：使用非阻塞通信
"""

import time
import rospy
from mpi4py import MPI

# ============= MPI标签常量定义 =============
# 定义所有MPI通信使用的标签，确保通信的唯一性和可维护性

TAG_ROBOT_STATE = 201              # 主控机器人状态（主进程→障碍物进程）
TAG_PEDESTRIAN_STATES = 202        # 所有行人状态字典（主进程→障碍物进程）
TAG_DYNAMIC_STATE = 200            # 动态障碍物自己的状态（障碍物进程→主进程）
TAG_CURRICULUM_UPDATE = 100        # 课程学习状态更新（主进程→障碍物进程）
TAG_TARGET_UPDATE = 300            # 目标点位置更新（主进程→目标发布进程）
TAG_SHUTDOWN = 999                 # 结束信号（主进程→所有进程）

# ============= 角色常量定义 =============
# 定义MPI进程的三种角色

ROLE_MAIN = "main_robot"                # 主控机器人（rank=0）
ROLE_TARGET_PUBLISHER = "target_publisher"  # 目标点发布机器人（rank=9）
ROLE_DYNAMIC_OBSTACLE = "dynamic_obstacle"  # 动态障碍物（rank=1-8）


class MPIHandler:
    """
    MPI通信管理器类

    职责：
        - 封装所有MPI通信逻辑
        - 提供统一的状态发送/接收接口
        - 管理进程角色识别
        - 处理课程学习状态广播

    属性：
        mpi_comm: MPI通信器（MPI.COMM_WORLD）
        rank: 当前进程的rank
        size: 总进程数
        role: 当前进程的角色
        robot_index: 机器人索引（仅对动态障碍物有意义）

    使用示例：
        ```python
        # 初始化
        mpi_handler = MPIHandler()

        # 获取角色
        role, robot_index = mpi_handler.get_role_and_index()

        # 发送状态（障碍物进程）
        mpi_handler.send_my_state_to_main(robot_id=1, position=[1.0, 2.0, 0.0], velocity=[0.1, 0.2])

        # 接收状态（主进程）
        states_dict = mpi_handler.receive_dynamic_states()
        ```
    """

    def __init__(self, comm=None, rank=None, size=None):
        """
        初始化MPI通信管理器

        参数：
            comm: MPI通信器，如果为None则使用MPI.COMM_WORLD
            rank: MPI进程rank，如果为None则自动获取
            size: MPI进程总数，如果为None则自动获取
        """
        # ============= MPI通信器初始化 =============
        if comm is None:
            self.mpi_comm = MPI.COMM_WORLD
        else:
            self.mpi_comm = comm

        # ============= 获取rank和size =============
        if rank is None:
            self.rank = self.mpi_comm.Get_rank()
        else:
            self.rank = rank

        if size is None:
            self.size = self.mpi_comm.Get_size()
        else:
            self.size = size

        # ============= 角色和索引识别 =============
        self.role, self.robot_index = self.get_role_and_index()

        rospy.loginfo(
            f"🔥 MPIHandler初始化完成: rank={self.rank}/{self.size}, "
            f"角色={self.role}, 机器人索引={self.robot_index}"
        )

    def get_role_and_index(self):
        """
        根据MPI rank确定进程角色和机器人索引

        返回：
            tuple: (role, robot_index)
                - role: str，角色标识（ROLE_MAIN | ROLE_TARGET_PUBLISHER | ROLE_DYNAMIC_OBSTACLE）
                - robot_index: int，机器人编号
                    - 主控机器人：0
                    - 目标发布机器人：
                    - 动态障碍物：1-8

        示例：
            ```python
            role, robot_index = mpi_handler.get_role_and_index()
            if role == ROLE_MAIN:
                # 启动主控训练逻辑
                pass
            elif role == ROLE_DYNAMIC_OBSTACLE:
                # 启动ORCA控制逻辑
                pass
            ```
        """
        if self.rank == 0:
            # 主控机器人（负责训练和控制robot_0）
            return (ROLE_MAIN, 0)

        elif self.rank == 9:
            # 目标点发布机器人（robot_9，用于可视化目标点）
            return (ROLE_TARGET_PUBLISHER, 9)

        elif 1 <= self.rank <= 8:
            # 动态障碍物（robot_1 到 robot_8）
            robot_index = self.rank
            return (ROLE_DYNAMIC_OBSTACLE, robot_index)

        else:
            # 未知rank，记录错误
            rospy.logerr(f"❌ MPIHandler: 未知的MPI rank: {self.rank}")
            return (None, None)

    def send_my_state_to_main(self, robot_id, position, velocity, active=True):
        """
        动态障碍物进程：将自己的状态发送给主进程

        此方法应由动态障碍物进程（rank 1-8）调用，用于将自己的位置和速度信息
        发送给主进程，以便主进程可以广播给其他障碍物进程进行ORCA协同避障。

        参数：
            robot_id (int): 机器人索引（1-8）
            position (list): [x, y, z] 位置，单位：米
            velocity (list): [vx, vy] 速度，单位：米/秒
            active (bool): 是否激活，默认True

        通信流程：
            障碍物进程 --[TAG_DYNAMIC_STATE]--> 主进程

        示例：
            ```python
            # 在ORCA控制循环中调用
            mpi_handler.send_my_state_to_main(
                robot_id=1,
                position=[px, py, 0.0],
                velocity=[vx, vy]
            )
            ```
        """
        # 只有动态障碍物进程才能调用此方法
        if self.rank == 0:
            rospy.logwarn("⚠️ 主进程不应调用send_my_state_to_main")
            return

        try:
            # 构造状态消息
            state_msg = {
                'robot_id': robot_id,
                'position': position,  # [x, y, z]
                'velocity': velocity,  # [vx, vy]
                'active': active,
                'timestamp': time.time()
            }

            # 非阻塞发送给主进程（rank=0）
            self.mpi_comm.isend(state_msg, dest=0, tag=TAG_DYNAMIC_STATE)

            # 每100次打印一次调试信息（避免日志过多）
            if not hasattr(self, '_send_counter'):
                self._send_counter = 0
            self._send_counter += 1

            if self._send_counter % 100 == 0:
                rospy.logdebug(
                    f"🔄 robot_{robot_id} MPI状态已发送: "
                    f"位置({position[0]:.2f}, {position[1]:.2f}), "
                    f"速度({velocity[0]:.2f}, {velocity[1]:.2f})"
                )

        except Exception as e:
            rospy.logwarn(f"❌ robot_{robot_id} MPI状态发送失败: {e}")

    def receive_dynamic_states(self):
        """
        主进程：从MPI接收所有动态障碍物的状态

        此方法应由主进程（rank=0）调用，用于接收所有动态障碍物进程发送的状态信息。
        采用非阻塞接收，避免等待。

        返回：
            dict: 动态障碍物状态字典，格式为：
                {
                    robot_id: {
                        'position': [x, y, z],
                        'velocity': [vx, vy],
                        'active': bool,
                        'last_update': float (timestamp)
                    },
                    ...
                }

        通信流程：
            主进程 <--[TAG_DYNAMIC_STATE]-- 障碍物进程

        使用场景：
            - 在step函数中调用，更新障碍物状态
            - 在广播前调用，获取最新状态

        示例：
            ```python
            # 在主进程的step函数中
            states_dict = mpi_handler.receive_dynamic_states()
            # states_dict现在包含所有障碍物的最新状态
            ```
        """
        # 只有主进程才能调用此方法
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用receive_dynamic_states")
            return {}

        try:
            # 非阻塞检查是否有消息
            while self.mpi_comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_DYNAMIC_STATE):
                # 接收状态消息
                state_msg = self.mpi_comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DYNAMIC_STATE)

                # 提取信息
                robot_id = state_msg['robot_id']
                position = state_msg['position']
                velocity = state_msg['velocity']
                active = state_msg.get('active', True)

                # 返回状态（外部会更新到环境的dynamic_obstacles_mpi_states）
                yield robot_id, position, velocity, active

        except Exception as e:
            rospy.logwarn(f"❌ MPI状态接收失败: {e}")

    def broadcast_robot_state(self, robot_state, total_dynamic=8):
        """
        主进程：广播主控机器人状态给所有动态障碍物进程

        此方法由主进程在step函数中调用，将主控机器人的状态广播给所有
        动态障碍物进程，用于ORCA避障计算。

        参数：
            robot_state (dict): 主控机器人状态，包含：
                - 'x': float，x坐标
                - 'y': float，y坐标
                - 'theta': float，朝向角
                - 'v': float，线速度
                - 'w': float，角速度
                - 'timestamp': float，时间戳
            total_dynamic (int): 动态障碍物总数，默认8

        通信流程：
            主进程 --[TAG_ROBOT_STATE]--> 所有障碍物进程

        示例：
            ```python
            # 在step函数中
            robot_state_msg = {
                'x': x, 'y': y, 'theta': theta,
                'v': v, 'w': w,
                'timestamp': time.time()
            }
            mpi_handler.broadcast_robot_state(robot_state_msg)
            ```
        """
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用broadcast_robot_state")
            return

        try:
            # 广播给所有动态障碍物进程（rank 1 到 total_dynamic）
            for rank in range(1, total_dynamic + 1):
                if rank < self.size:  # 确保目标rank存在
                    self.mpi_comm.isend(robot_state, dest=rank, tag=TAG_ROBOT_STATE)

        except Exception as e:
            rospy.logwarn_throttle(10, f"❌ 广播主控机器人状态失败: {e}")

    def broadcast_all_pedestrians(self, pedestrians_dict, total_dynamic=8):
        """
        主进程：广播所有行人状态字典给所有动态障碍物进程

        此方法由主进程在step函数中调用，将所有动态障碍物的状态字典广播给
        每个动态障碍物进程，用于ORCA多行人协同避障。

        参数：
            pedestrians_dict (dict): 所有行人状态字典，格式与receive_dynamic_states相同
            total_dynamic (int): 动态障碍物总数，默认8

        通信流程：
            主进程 --[TAG_PEDESTRIAN_STATES]--> 所有障碍物进程

        示例：
            ```python
            # 在step函数中
            mpi_handler.broadcast_all_pedestrians(self.dynamic_obstacles_mpi_states)
            ```
        """
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用broadcast_all_pedestrians")
            return

        try:
            # 广播给所有动态障碍物进程
            for rank in range(1, total_dynamic + 1):
                if rank < self.size:
                    self.mpi_comm.isend(pedestrians_dict, dest=rank, tag=TAG_PEDESTRIAN_STATES)

        except Exception as e:
            rospy.logwarn_throttle(10, f"❌ 广播行人状态失败: {e}")

    def broadcast_curriculum_state(self, active_dynamic_count, total_dynamic=8):
        """
        主进程：广播课程学习状态给所有动态障碍物进程

        此方法由主进程调用，用于通知动态障碍物进程当前激活的障碍物数量，
        从而实现课程学习（逐步增加障碍物数量的训练策略）。

        参数：
            active_dynamic_count (int): 当前激活的动态障碍物数量
            total_dynamic (int): 动态障碍物总数，默认8

        通信流程：
            主进程 --[TAG_CURRICULUM_UPDATE]--> 所有障碍物进程

        消息格式：
            {
                "command": "curriculum_update",
                "active_dynamic": int,
                "timestamp": float
            }

        示例：
            ```python
            # 在init_main_robot_obstacles_only中调用
            # 启动时全部激活（方案A）
            mpi_handler.broadcast_curriculum_state(active_dynamic_count=8)
            ```
        """
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用broadcast_curriculum_state")
            return

        try:
            # 构造课程学习消息
            curriculum_msg = {
                "command": "curriculum_update",
                "active_dynamic": active_dynamic_count,
                "timestamp": time.time()
            }

            # 广播给所有动态障碍物进程
            for rank in range(1, total_dynamic + 1):
                if rank < self.size:
                    self.mpi_comm.send(curriculum_msg, dest=rank, tag=TAG_CURRICULUM_UPDATE)

            rospy.loginfo(
                f"🔥 课程学习状态已广播: 激活{active_dynamic_count}/{total_dynamic}个动态障碍物"
            )

        except Exception as e:
            rospy.logwarn(f"❌ 广播课程学习状态失败: {e}")

    def send_target_update(self, target_position, dest_rank=9):
        """
        主进程：发送目标点位置更新给目标点发布进程

        此方法由主进程在目标点更新时调用，将新的目标点位置发送给目标点发布进程（robot_45），
        用于可视化显示。

        参数：
            target_position (list): [x, y] 目标点位置
            dest_rank (int): 目标进程的rank，默认9（robot_45）

        通信流程：
            主进程 --[TAG_TARGET_UPDATE]--> 目标发布进程

        示例：
            ```python
            # 在GenerateTargetPoint或GenerateTargetPoint_test中调用
            mpi_handler.send_target_update(self.target_point)
            ```
        """
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用send_target_update")
            return

        try:
            # 构造目标点更新消息
            target_msg = {
                "command": "target_update",
                "target_position": target_position,
                "timestamp": time.time()
            }

            # 发送给目标点发布进程
            if dest_rank < self.size:
                self.mpi_comm.send(target_msg, dest=dest_rank, tag=TAG_TARGET_UPDATE)
                rospy.logdebug(f"🎯 目标点已发送给robot_45: {target_position}")

        except Exception as e:
            rospy.logwarn(f"❌ 发送目标点更新失败: {e}")

    def send_shutdown_signal(self, total_processes=10):
        """
        主进程：发送结束信号给所有其他进程

        此方法由主进程在训练结束或异常退出时调用，通知所有其他进程优雅退出。

        参数：
            total_processes (int): 总进程数，默认10

        通信流程：
            主进程 --[TAG_SHUTDOWN]--> 所有进程
        """
        if self.rank != 0:
            rospy.logwarn("⚠️ 只有主进程可以调用send_shutdown_signal")
            return

        try:
            shutdown_msg = {"command": "shutdown", "timestamp": time.time()}

            for rank in range(1, total_processes):
                if rank < self.size:
                    self.mpi_comm.send(shutdown_msg, dest=rank, tag=TAG_SHUTDOWN)

            rospy.loginfo("🛑 结束信号已发送给所有进程")

        except Exception as e:
            rospy.logwarn(f"❌ 发送结束信号失败: {e}")

    def check_shutdown_signal(self):
        """
        非主进程：检查是否收到结束信号

        此方法由非主进程（动态障碍物、目标点发布）定期调用，检查是否收到
        主进程发送的结束信号。

        返回：
            bool: 如果收到结束信号返回True，否则返回False

        示例：
            ```python
            # 在控制循环中
            while not rospy.is_shutdown():
                if mpi_handler.check_shutdown_signal():
                    rospy.loginfo("收到结束信号，退出控制循环")
                    break
                # ... 控制逻辑 ...
            ```
        """
        if self.rank == 0:
            return False  # 主进程不检查

        try:
            # 非阻塞检查是否有结束信号
            if self.mpi_comm.Iprobe(source=0, tag=TAG_SHUTDOWN):
                shutdown_msg = self.mpi_comm.recv(source=0, tag=TAG_SHUTDOWN)
                rospy.loginfo(f"🛑 robot_{self.robot_index} 收到结束信号")
                return True

        except Exception as e:
            rospy.logwarn(f"❌ 检查结束信号失败: {e}")

        return False

    def __repr__(self):
        """字符串表示"""
        return (
            f"MPIHandler(rank={self.rank}, size={self.size}, "
            f"role={self.role}, robot_index={self.robot_index})"
        )


# ============= 工具函数 =============

def create_mpi_handler():
    """
    工厂函数：创建MPIHandler实例

    返回：
        MPIHandler: 已初始化的MPI处理器实例
    """
    return MPIHandler()


def get_process_info(mpi_handler):
    """
    获取当前进程的详细信息

    参数：
        mpi_handler: MPIHandler实例

    返回：
        dict: 进程信息字典
    """
    return {
        'rank': mpi_handler.rank,
        'size': mpi_handler.size,
        'role': mpi_handler.role,
        'robot_index': mpi_handler.robot_index,
        'is_main': (mpi_handler.rank == 0),
        'is_obstacle': (1 <= mpi_handler.rank <= 8),
        'is_target_publisher': (mpi_handler.rank == 9)
    }
