import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random
from collections import deque
from scipy.stats import truncnorm
import signal
import std_srvs.srv
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, Point, Pose, Pose2D
from std_msgs.msg import Int8

# ============= 导入新创建的模块 =============
from utils.mpi_utils.mpi_handler import MPIHandler
from utils.dyn_obstacle_state_manager import DynamicObstacleStateManager
from utils.collision_detector import CollisionDetector

class StageWorld():
    """
    机器人导航环境类 (固定网格课程学习版本)
    ===========================================
    
    �� SOTA特性：
    1. 6×6固定网格系统，确保障碍物位置的一致性和可重现性
    2. 并行课程学习，同时调整静态和动态障碍物难度
    3. 智能化的课程升级机制，基于测试性能自动调节
    4. 高效的障碍物管理，减少计算开销
    """
    
    def __init__(self, beam_num, index=0, num_env=1):
        """
        初始化机器人环境 (MPI多进程版本)
        
        Args:
            beam_num (int): 激光雷达光束数量（通常为720）
            index (int): MPI进程索引，对应机器人编号
            num_env (int): 环境总数
        """
        #! ============= �� MPI多进程机器人配置 �� =============
        self.robot_index = index
        self.num_env = num_env
        self.is_main_robot = (index == 0)  # 只有robot_0是主控机器人
        
        print(f"�� 初始化机器人环境: robot_{index} ({'主控' if self.is_main_robot else '动态障碍物'})")
        
        # ============= ROS节点初始化 =============
        node_name = f'StageWorld_robot_{index}'
        rospy.init_node(node_name, anonymous=False)
        print(f"�� ROS节点已创建: {node_name}")
        
        # ============= 加载测试数据集（增强版本） =============
        # goal_set1 = np.load('dataset/goal_set_sz_item.npy')      
        # robot_set1 = np.load('dataset/robot_set_sz_item.npy')    
        config_set1 = np.load('dataset/config_set_nev1.npy') 
        goal_set = np.load('map/goal_set_nev1.npy')
        robot_set = np.load('map/robot_set_nev1.npy')  
        # goal_set1 = np.load('dataset/goal_set277.npy')
        # robot_set1 = np.load('dataset/robot_set277.npy')  

        self.test_targets = goal_set         
        self.test_initials = robot_set       
        self.config_initials = config_set1   

        # ============= 基础环境参数 =============
        self.move_base_goal = PoseStamped()        
        self.image_size = [224, 224]               
        self.bridge = CvBridge()                   

        # 机器人状态相关变量
        self.object_state = [0, 0, 0, 0]                             
        self.stalled = False                       
        self.crash_stop = False                    

        # 机器人运动参数
        self.self_speed = [0.3, 0.0]                              
        
        # 历史动作缓存（用于动作平滑）
        self.past_actions = deque(maxlen=2)        
        for initial_zero in range(2):
            self.past_actions.append(0)           

        # 时间和步数控制
        self.start_time = time.time()              
        self.max_steps = 10000                     
        self.gap = 0.5                            
        self.last_position = [0.0, 0.0, 0.0]     

        # ============= 激光雷达相关参数 =============
        self.scan = None                          
        self.beam_num = beam_num                  
        self.laser_cb_num = 0                                        

        # ============= 环境和控制参数 =============                                       
        self.step_target = [0., 0.]                                  
        self.stop_counter = 0                     

        # ============= 动作空间定义 =============
        self.max_action = [0.7, np.pi/2]         
        self.max_acc = [2.0, 2.0]                      
        self.ratio = 1.0                          

        # 初始化机器人速度
        self.self_speed = [0.3/self.ratio, 0.0]
        self.target_point = [0, 5.5]             

        #! ============= �� 新的固定网格课程学习系统 �� =============
        #? change by shanze 1014
        self.total_static = 0                     #! 静态障碍物总数设置为0
        self.total_dynamic = 8                    #! 动态障碍物总数设置为8
      
        self.curriculum_levels = [0] # 无静态障碍物
        self.dynamic_curriculum_levels = [2] 
        
        #? change by shanze 1014
        self.current_curriculum_level = 0
        self.current_dynamic_level = 2 # 默认一直激活这1个动态障碍物
        self.active_static_num = 0
        self.active_dynamic_num = 2
        
        #! 课程学习区域参数
        self.area_center_x = 0.0                #! 障碍物运动区域中心x坐标
        self.area_size = 20.0                     #! 地图实际大小20×20
        self.area_half = self.area_size / 2.0 - 1.0  #! 障碍物运动区域半径
        self.human_radius = 0.3                   #! 障碍物半径
        self.discomfort_dist = 0.5                #! 舒适距离
        self.v_pref = 0.6                         #! 障碍物首选速度
        self.update_rate = 10                     #! 障碍物更新频率

        # ============= 地图处理 =============
        map_img = cv2.imread('map/d6.jpg', 0)
        ret, binary_map = cv2.threshold(map_img, 10, 1, cv2.THRESH_BINARY)  
        binary_map = 1 - binary_map               
        self.map = binary_map.astype(np.float32)  
        
        height, width = binary_map.shape
        self.map_pixel = np.array([width, height])  

        # ============= 多尺度地图配置 =============
        # TODO 暂时移除，固定地图尺寸
        self.map_sizes = np.zeros((8, 2))
        for map_no in range(8):
            self.map_sizes[map_no, 0] = 20.0#0.0 #! * (0.8**map_no)  
            self.map_sizes[map_no, 1] = 20.0#10.0 #! * (0.8**map_no)  
        
        self.map_origin = self.map_pixel / 2 - 1   

        self.map_size = self.map_sizes[0]
        self.R2P = self.map_pixel / self.map_size

        # ============= 机器人物理参数 =============
        self.robot_size = 0.4
        self.target_size = 0.4

        self.robot_range_x1 = 0.4
        self.robot_range_x2 = 0.4
        self.robot_range = 0.2
        self.robot_range_y = 0.4

        # ============= Pool策略配置（用于切换最近点/双点模式）=============
        # False: 方案A - 最近点策略，输出 [90, 9]，state维度 = 90*9 + 8 = 818
        # True:  方案B - 双点策略，输出 [180, 9]，state维度 = 180*9 + 8 = 1628
        self.use_dual_point_pool = True                  

        # ============= 地图中心点配置 =============
        self.map_center = np.zeros((9, 2))
        for map_no in range(8):
            # ? 地图中心点配置 -40 0 ->0 0 
            self.map_center[map_no, 0] = 0.0
            self.map_center[map_no, 1] = 0.0
        
        self.env = 0                              
        self.control_period = 0.2                 

        # ============= �� MPI多进程ROS发布者和订阅者 �� =============
        # 每个MPI进程连接到对应编号的机器人
        cmd_vel_topic = f'/robot_{self.robot_index}/cmd_vel'
        cmd_pose_topic = f'/robot_{self.robot_index}/cmd_pose'
        laser_topic = f'/robot_{self.robot_index}/base_scan'
        object_state_topic = f'/robot_{self.robot_index}/base_pose_ground_truth'
        odom_topic = f'/robot_{self.robot_index}/odom'
        stall_topic = f'/robot_{self.robot_index}/stalled'
        
        #! �� 目标点发布器：主控机器人和robot_9都需要
        if self.is_main_robot or self.robot_index == 9:
            if self.robot_index == 9:
                target_topic = f'/robot_{self.robot_index}/cmd_pose'
            else:
                # 主控机器人通过robot_9的话题发布目标点
                target_topic = f'/robot_9/cmd_pose'
            self.pose_publish_goal = rospy.Publisher(target_topic, Pose2D, queue_size=1000)
            rospy.loginfo(f"�� 目标点发布器已创建: {target_topic}")

        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=100)         
        self.pose_publisher = rospy.Publisher(cmd_pose_topic, Pose2D, queue_size=1000)  
        rospy.loginfo(f"�� 发布者已创建: {cmd_pose_topic}")
        
        #! ============= �� MPI架构：每个进程只订阅自己的话题 �� =============
        # 每个进程只订阅自己机器人的状态信息
        self.object_state_sub = rospy.Subscriber(
            object_state_topic, Odometry, self.GroundTruthCallBack
        )  
        
        #! 只有主控机器人需要激光雷达和里程计数据
        if self.is_main_robot:
            self.laser_sub = rospy.Subscriber(
                laser_topic, LaserScan, self.LaserScanCallBack
            )  
            self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.OdometryCallBack)  
            self.sim_clock = rospy.Subscriber('clock', Clock, self.SimClockCallBack)   
            self.ResetStage = rospy.ServiceProxy('reset_positions', std_srvs.srv.Empty)  
            self.stalls = rospy.Subscriber(stall_topic, Int8, self.update_robot_stall_data)
        

        rospy.loginfo(f"�� MPI架构ROS通信设置完成: robot_{self.robot_index}")

        #! ============= �� MPI通信设置 �� =============
        # 创建MPI处理器（将在训练文件中通过set_mpi_handler设置）
        self.mpi_handler = None

        # 动态障碍物状态缓存（通过MPI更新）
        self.dynamic_obstacles_mpi_states = {}

        for i in range(self.total_dynamic):
            robot_id = i + 1  # robot_1 到 robot_8
            self.dynamic_obstacles_mpi_states[robot_id] = {
                'position': [0.0, 0.0, 0.0],
                'velocity': [0.0, 0.0],
                'last_update': time.time(),
                'active': False
            }

        # 创建动态障碍物状态管理器（支持最多5个障碍物）
        self.obstacle_manager = DynamicObstacleStateManager(max_dynamic_num=5)

        # 🔥 初始化碰撞检测器为None（将在ResetWorld或reset时延迟创建）
        # 原因：length1, length2, width等属性在ResetWorld()中才设置
        # 在ResetWorld()和set_robot_pose_test()中会创建实际的CollisionDetector实例
        self.collision_detector = None

        # ============= 等待初始化完成 =============
        # �� 只有主控机器人需要等待激光雷达数据
        if self.is_main_robot:
            while self.scan is None:
                rospy.sleep(0.1)  # 添加短暂睡眠避免CPU占用过高
                if rospy.is_shutdown():
                    break
            rospy.loginfo("�� 主控机器人激光雷达数据已就绪")
        else:
            # �� 非主控机器人不需要激光雷达数据，直接继续
            rospy.loginfo(f"�� 非主控机器人 robot_{self.robot_index} 跳过激光雷达等待")
        
        rospy.on_shutdown(self.shutdown)

        rospy.sleep(1.)

    
    def publish_target_point(self, target_point):
        """
        发布目标点用于可视化（控制绿色target_marker移动）

        功能说明：
            - 将target_point位置发布到/robot9/cmd_pose topic
            - 控制Stage world中的target_marker（绿色块）移动
            - 在训练和测试时都会被调用
        """
        if not hasattr(self, 'target_point') or self.target_point is None:
            return

        if not hasattr(self, 'pose_publish_goal'):
            rospy.logwarn_once("pose_publish_goal publisher未初始化，无法发布目标点")
            return

        # 构造Pose2D消息
        target_pose = Pose2D()
        target_pose.x = target_point[0]
        target_pose.y = target_point[1]
        target_pose.theta = 0.0

        # 发布到/robot9/cmd_pose topic，控制target_marker移动
        self.pose_publish_goal.publish(target_pose)

        # 调试日志（每20次发布打印一次）
        if not hasattr(self, '_target_publish_counter'):
            self._target_publish_counter = 0

        # self._target_publish_counter += 1
        # if self._target_publish_counter % 20 == 0:
        #     rospy.loginfo_throttle(2, f"目标点已发布: ({target_pose.x:.2f}, {target_pose.y:.2f})")
 

    def GroundTruthCallBack(self, GT_odometry):
        """真实位姿回调函数"""
        Quaternions = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        
        self.state_GT = [
            GT_odometry.pose.pose.position.x,
            GT_odometry.pose.pose.position.y,
            Euler[2],  
        ]
        
        v_x = GT_odometry.twist.twist.linear.x   
        v_y = GT_odometry.twist.twist.linear.y   
        v = np.sqrt(v_x**2 + v_y**2)             
        
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]


    def update_robot_stall_data(self, msg):
        """更新机器人卡住状态"""
        self.stalled = msg.data


    def set_robot_pose(self):
        """随机设置机器人位置"""
        robot_pose_data = Pose2D()
        
        x = random.uniform(
            -(self.map_size[0] / 2 - self.target_size),
            self.map_size[0] / 2 - self.target_size,
        )
        y = random.uniform(
            -(self.map_size[1] / 2 - self.target_size),
            self.map_size[1] / 2 - self.target_size,
        )

        # 循环生成随机位置，直到找到既无静态障碍物碰撞，也与动态障碍物保持安全距离的位置
        while (not self.robotPointCheck(x, y) or
               not self.dynamicObstaclePointCheck(x, y)) and not rospy.is_shutdown():
            x = random.uniform(
                -(self.map_size[0] / 2 - self.target_size),
                self.map_size[0] / 2 - self.target_size,
            )
            y = random.uniform(
                -(self.map_size[1] / 2 - self.target_size),
                self.map_size[1] / 2 - self.target_size,
            )
        
        robot_pose_data.x = x + self.map_center[self.env, 0]
        robot_pose_data.y = y + self.map_center[self.env, 1]
        robot_pose_data.theta = 0.0
        self.pose_publisher.publish(robot_pose_data)


    def targetPointCheck(self, x, y):
        """检查目标点位置是否有效"""
        target_x = x
        target_y = y
        pass_flag = True
        
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        
        window_size = int(self.target_size * np.amax(self.R2P))
        
        for x in range(
            np.amax([0, x_pixel - window_size]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def robotPointCheck(self, x, y):
        """检查机器人位置是否有效"""
        target_x = x
        target_y = y
        pass_flag = True
        
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        
        window_size_x1 = int(self.robot_range_x1 * np.amax(self.R2P))  
        window_size_x2 = int(self.robot_range_x2 * np.amax(self.R2P))  
        window_size_y = int(self.robot_range_y * np.amax(self.R2P))    
        
        for x in range(
            np.amax([0, x_pixel - window_size_x2]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size_x1]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size_y]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size_y]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def pedestrianPointCheck(self, x, y):
        """检查机器人位置是否有效"""
        target_x = x
        target_y = y
        pass_flag = True
        
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        
        window_size_x1 = int((self.robot_range_x1 + 0.15) * np.amax(self.R2P))  
        window_size_x2 = int((self.robot_range_x2 + 0.15) * np.amax(self.R2P))  
        window_size_y = int((self.robot_range_y + 0.15) * np.amax(self.R2P))    
        
        for x in range(
            np.amax([0, x_pixel - window_size_x2]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size_x1]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size_y]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size_y]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def dynamicObstaclePointCheck(self, x, y):
        """
        检查候选位置与动态障碍物的距离是否安全

        检测逻辑：
        1. 遍历所有激活的动态障碍物（行人）
        2. 计算候选位置与每个障碍物的中心距离
        3. 将矩形机器人近似为圆形（外接圆半径）
        4. 计算净间隙：distance - robot_radius - obstacle_radius
        5. 要求净间隙 > 0.5m 才判定为安全

        参数：
            x (float): 候选机器人位置x坐标（世界坐标系，未加地图中心偏移）
            y (float): 候选机器人位置y坐标（世界坐标系，未加地图中心偏移）

        返回：
            bool: True=安全位置可生成，False=与动态障碍物过近不可生成

        设计说明：
            - 机器人等效半径：sqrt((length1+length2)²/4 + width²/4) - 矩形外接圆半径
            - 行人半径：0.3m - 圆柱体半径
            - 安全间隙要求：> 0.5m - 用户指定的最小净间隙
            - 陈旧数据：超过5秒未更新的障碍物数据会被自动忽略
        """
        # ============= 步骤1：数据有效性检查 =============
        # 检查MPI状态字典是否存在（防止初始化阶段调用）
        if not hasattr(self, 'dynamic_obstacles_mpi_states'):
            return True  # 字典未初始化，无障碍物需要检查

        if not self.dynamic_obstacles_mpi_states:
            return True  # 字典为空，无障碍物需要检查

        # ============= 步骤2：计算机器人等效半径 =============
        # 将矩形机器人近似为圆形：计算能够包含整个矩形的最小圆（外接圆）的半径
        # 公式推导：矩形的对角线长度 = sqrt((length1+length2)² + width²)
        #          外接圆半径 = 对角线长度 / 2
        # 简化为：robot_radius = sqrt((length1+length2)²/4 + width²/4)
        robot_radius = math.sqrt(
            ((self.length1 + self.length2) / 2.0) ** 2 +
            (self.width / 2.0) ** 2
        )

        # ============= 步骤3：定义常量 =============
        OBSTACLE_RADIUS = 0.3        # 行人圆柱体半径（米）
        SAFETY_CLEARANCE = 0.5       # 用户要求的最小净间隙（米）
        STALENESS_THRESHOLD = 5.0    # 数据有效期阈值（秒）

        # ============= 步骤4：遍历所有动态障碍物进行安全检查 =============
        current_time = time.time()  # 获取当前时间戳用于判断数据新鲜度

        for robot_id, obstacle_state in self.dynamic_obstacles_mpi_states.items():
            # --- 4.1 激活状态检查 ---
            # 跳过未激活的障碍物（避免检查尚未启动的行人进程）
            if not obstacle_state.get('active', False):
                continue

            # --- 4.2 数据新鲜度检查 ---
            # 跳过陈旧数据（超过5秒未更新，可能是进程已停止或通信中断）
            last_update = obstacle_state.get('last_update', 0)
            if (current_time - last_update) >= STALENESS_THRESHOLD:
                continue  # 数据过旧，忽略该障碍物

            # --- 4.3 提取障碍物位置 ---
            # position格式：[x, y, z]，世界坐标系（已包含地图中心偏移）
            position = obstacle_state.get('position', [0.0, 0.0, 0.0])
            px = position[0]  # 障碍物x坐标
            py = position[1]  # 障碍物y坐标
            # 注意：position已经是世界坐标系（含地图中心偏移），
            #      而候选位置(x, y)是局部坐标（未含偏移），
            #      因此需要将候选位置转换为世界坐标进行比较

            # --- 4.4 坐标系转换 ---
            # 将候选位置从局部坐标转换为世界坐标（加上地图中心偏移）
            world_x = x + self.map_center[self.env, 0]
            world_y = y + self.map_center[self.env, 1]

            # --- 4.5 计算中心距离 ---
            # 欧几里得距离：sqrt((x1-x2)² + (y1-y2)²)
            distance = math.sqrt((world_x - px) ** 2 + (world_y - py) ** 2)

            # --- 4.6 计算净间隙 ---
            # 净间隙 = 中心距离 - 机器人半径 - 障碍物半径
            # 这是两个圆形物体表面之间的实际空隙距离
            clearance = distance - robot_radius - OBSTACLE_RADIUS

            # --- 4.7 安全性判定 ---
            # 如果净间隙 <= 0.5m，则判定为不安全，拒绝该位置
            if clearance <= SAFETY_CLEARANCE:
                # 可选：输出调试信息（生产环境可注释掉）
                # rospy.logdebug(
                #     f"[动态障碍物检测] 位置被拒绝: ({x:.2f}, {y:.2f}) "
                #     f"距离障碍物{robot_id}@({px:.2f}, {py:.2f}) "
                #     f"净间隙{clearance:.2f}m <= {SAFETY_CLEARANCE}m"
                # )
                return False  # 不安全，拒绝该位置

        # ============= 步骤5：所有检查通过 =============
        # 所有激活且有效的障碍物都满足安全距离要求
        return True  # 安全位置，可以生成机器人

    def LaserScanCallBack(self, scan):
        """激光雷达数据回调函数"""
        self.scan_param = [
            scan.angle_min,        
            scan.angle_max,        
            scan.angle_increment,  
            scan.time_increment,   
            scan.scan_time,        
            scan.range_min,        
            scan.range_max,        
        ]
        
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1  

    def OdometryCallBack(self, odometry):
        """里程计数据回调函数"""
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w]
        )
        
        self.state = [
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            Euler[2],
        ]
        
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def SimClockCallBack(self, clock):
        """仿真时钟回调函数"""
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.



    def GetNoisyLaserObservation(self):
        """获取带噪声的激光雷达观测"""
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 2.0  
        
        nuniform_noise = np.random.uniform(-0.01, 0.01, scan.shape)
        linear_noise = np.multiply(np.random.normal(0., 0.01, scan.shape), scan)
        
        noise = nuniform_noise + linear_noise
        scan += noise
        
        scan[scan < 0.] = 0.
        
        return scan

    def GetSelfState(self):
        """获取机器人估计状态"""
        return self.state

    def GetSelfStateGT(self):
        """获取机器人真实状态"""
        return self.state_GT

    def GetSelfSpeedGT(self):
        """获取机器人真实速度"""
        return self.speed_GT

    def match_lidar_with_dynamic_obstacles(self, lidar_data, robot_state, dynamic_states):
        """
        将动态障碍物的ground truth位置与lidar点进行匹配（改进版）

        通过动态障碍物的绝对位置，计算其在lidar坐标系中的角度范围，
        找到对应的lidar点并标记为动态障碍物，附加速度信息。

        改进点：
        1. 使用相对距离容差（max(0.2, 0.15 * expected_distance)）替代固定50cm容差
        2. 处理多障碍物优先级：只有当新障碍物更近时才更新

        参数：
            lidar_data: [540] 原始lidar距离数据
            robot_state: 机器人状态 [x, y, theta]
            dynamic_states: 动态障碍物状态字典 {robot_id: {'position': [px, py, pz], 'velocity': [vx, vy], ...}}

        返回：
            matched_velocities: [540, 2] 每个lidar点对应的速度 (vx, vy)，机器人坐标系
            is_dynamic: [540] bool数组，标记该点是否属于动态障碍物
        """
        matched_velocities = np.zeros((540, 2), dtype=np.float32)
        is_dynamic = np.zeros(540, dtype=bool)
        # 新增：记录每个lidar点匹配到的动态障碍物距离，用于处理多障碍物优先级
        dynamic_distances = np.full(540, np.inf, dtype=np.float32)

        x, y, theta = robot_state

        # Lidar参数：360度，540个点
        angle_min = -np.pi  # -180度
        angle_increment = 2 * np.pi / 540  # 每个点的角度增量

        for robot_id, obs_state in dynamic_states.items():
            if not obs_state.get('active', False):
                continue

            # 获取障碍物世界坐标
            pos = obs_state.get('position', [0, 0, 0])
            vel = obs_state.get('velocity', [0, 0])
            px, py = pos[0], pos[1]
            vx_world, vy_world = vel[0], vel[1]
            radius = 0.3  # roomba半径 (world文件中point定义的实际半径)

            # 转换到机器人坐标系
            dx = px - x
            dy = py - y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.1:  # 太近，跳过
                continue

            # 相对角度（机器人坐标系）
            relative_angle = np.arctan2(dy, dx) - theta
            # 归一化到 [-pi, pi]
            relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

            # 速度转换到机器人坐标系
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            vx_local = vx_world * cos_t + vy_world * sin_t
            vy_local = -vx_world * sin_t + vy_world * cos_t

            # 障碍物覆盖的角度范围（由于有半径）
            half_angle = np.arcsin(min(radius / distance, 1.0))

            # 找到对应的lidar索引范围
            for i in range(540):
                beam_angle = angle_min + i * angle_increment
                # 归一化beam_angle到[-pi, pi]
                beam_angle = np.arctan2(np.sin(beam_angle), np.cos(beam_angle))

                # 检查该beam是否在障碍物角度范围内
                angle_diff = abs(beam_angle - relative_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                if angle_diff <= half_angle:
                    # 额外验证：检查lidar测量距离是否与预期距离匹配
                    expected_distance = distance - radius  # 减去半径
                    measured_distance = lidar_data[i]

                    # 改进1：使用相对容差（最小0.2m，或者15%的期望距离）
                    distance_tolerance = max(0.2, 0.15 * expected_distance)

                    if abs(measured_distance - expected_distance) < distance_tolerance:
                        # 改进2：只有当新障碍物更近时才更新（处理多障碍物优先级）
                        if distance < dynamic_distances[i]:
                            matched_velocities[i] = [vx_local, vy_local]
                            is_dynamic[i] = True
                            dynamic_distances[i] = distance

        return matched_velocities, is_dynamic

    def GetSelfSpeed(self):
        """获取机器人估计速度"""
        return self.speed

    def GetSimTime(self):
        """获取仿真时间"""
        return self.sim_time

    def ResetWorld(self, env_no, length1, length2, width):
        """重置环境世界（完整重置）"""
        rospy.sleep(3.0)  
        
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        
        # TODO 降低了线速度
        # ============= 随机化动力学参数 =============
        # self.max_action[0] = np.random.uniform(0.3, 2.0)      
        # self.max_action[1] = np.random.uniform(np.pi/6.0, 2*np.pi)  
        # ? change by sz 1030 to test the effect of the action space
        self.max_action[0] = np.random.uniform(0.3, 1.0)      
        # self.max_action[1] = np.random.uniform(np.pi/6.0, 2*np.pi)  
        self.max_action[1] = np.random.uniform(np.pi/6, 2*np.pi)  
        
        self.max_acc[0] = np.random.uniform(0.5, 5.0)         
        self.max_acc[1] = np.random.uniform(np.pi/6.0, 2*np.pi)    
        
        self.length1 = length1
        self.length2 = length2
        self.width = width

        self.robot_range_x1 = self.length1 + 0.15
        self.robot_range_x2 = self.length2 + 0.15
        self.robot_range_y = self.width + 0.15

        # 🔥 创建或更新碰撞检测器（现在机器人尺寸参数已可用）
        if self.collision_detector is None:
            # 首次创建
            self.collision_detector = CollisionDetector(
                robot_length1=self.length1,
                robot_length2=self.length2,
                robot_width=self.width,
                pedestrian_radius=getattr(self, 'human_radius', 0.3),
                collision_mode='rectangle',
                safety_margin=0.0,
                state_timeout=5.0
            )
            rospy.loginfo("✅ CollisionDetector在ResetWorld中创建完成")
        else:
            # 更新现有实例的机器人尺寸
            self.collision_detector.update_robot_size(self.length1, self.length2, self.width)
            # rospy.loginfo("✅ CollisionDetector机器人尺寸已更新")

        self.stop_counter = 0.0
        self.crash_stop = False
        
        self.env = env_no
        self.map_size = self.map_sizes[env_no]  
        
        # ? change by sz, now the map size is fixed, so the target size is fixed
        self.target_size = 0.4     
        
        self.R2P = self.map_pixel / self.map_size
        
        self.set_robot_pose()
        
        self.stalls                    
        self.self_speed = [0.0, 0.0]   
        self.step_target = [0., 0.]              
        self.ratio = 1.0               
        self.start_time = time.time()   
        
        rospy.sleep(2.0)  
        return self.max_action[0]

    def Reset(self, env_no):
        """重置环境（简化版本）"""
        rospy.sleep(3.0)
        
        self.past_actions = deque(maxlen=2)
        for initial_zero in range(2):
            self.past_actions.append(0)
        
        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]

        # ? change by sz, now the map size is fixed, so the target size is fixed
        self.target_size = 0.4 
        
        self.R2P = self.map_pixel / self.map_size
        
        self.stalls
        self.self_speed = [0.0, 0.0]
        self.step_target = [0., 0.]
        self.ratio = 1.0
        self.start_time = time.time()

        rospy.sleep(2.0)
        return self.max_action[0]




    def set_robot_pose_test(self, i, env_no, robot_no):
        """设置测试用的机器人位置和配置"""
        # ? change by sz 1104 to reset episode counter
        # self._reset_episode_counter()  # ← 添加这一行
        self.max_action[0] = self.config_initials[robot_no, i, 3] * 0.5          
        self.max_action[1] = self.config_initials[robot_no, i, 4] * 2       
        
        self.max_acc[0] = self.config_initials[robot_no, i, 5]              
        self.max_acc[1] = self.config_initials[robot_no, i, 6]              
        
        # TODO 暂时移除，固定机器人尺寸
        # self.length1 = self.config_initials[robot_no, i, 0]
        # self.length2 = self.config_initials[robot_no, i, 1]
        # self.width = self.config_initials[robot_no, i, 2]
        # ? change by ym 1022, keep robot size
        self.length1 = 0.10
        self.length2 = 0.10
        self.width = 0.10

        # 🔥 创建或更新碰撞检测器（测试模式下也需要）
        if self.collision_detector is None:
            # 首次创建
            self.collision_detector = CollisionDetector(
                robot_length1=self.length1,
                robot_length2=self.length2,
                robot_width=self.width,
                pedestrian_radius=getattr(self, 'human_radius', 0.3),
                collision_mode='rectangle',
                safety_margin=0.0,
                state_timeout=5.0
            )
            rospy.loginfo("✅ CollisionDetector在set_robot_pose_test中创建完成")
        else:
            # 更新现有实例的机器人尺寸
            self.collision_detector.update_robot_size(self.length1, self.length2, self.width)

        self.stop_counter = 0.0
        self.crash_stop = False
        self.env = env_no
        self.map_size = self.map_sizes[env_no]
        self.stalls
        
        robot_pose_data = Pose2D()
        
        x = self.test_initials[robot_no, i, 0] * (1.25**(6)) 
        y = self.test_initials[robot_no, i, 1] * (1.25**(6)) 
        # ! 环境地图大小不变
        # x = self.test_initials[robot_no, 0]  #? * (0.75**(6-3))
        # y = self.test_initials[robot_no, 1]  #? * (0.75**(6-3))
        
        robot_pose_data.theta = 0.0
        robot_pose_data.x = x
        robot_pose_data.y = y
        
        self.pose_publisher.publish(robot_pose_data)
        
        rospy.sleep(2.)
        return self.max_action[0]

    def GenerateTargetPoint_test(self, i, env_no, robot_no):
        """生成测试用目标点"""
        self.env = env_no

        # TODO 暂时移除，固定目标点尺寸
        self.target_point = self.test_targets[robot_no, i, :] * (1.25**6)
        # self.target_point = self.test_targets[robot_no, i, :] #? * (0.75**(6-3))   #? change by sz 1027
        x = self.target_point[0]
        y = self.target_point[1]    
        
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

        # ! publish target
        self.publish_target_point(self.target_point)
        # === 新增：通过MPI同步目标点给robot_9 ===
        if hasattr(self, 'mpi_comm') and self.mpi_comm is not None:
            try:
                target_msg = {
                    "command": "target_update",
                    "target_position": self.target_point
                }
                self.mpi_comm.send(target_msg, dest=9, tag=300)
            except Exception as e:
                rospy.logwarn(f"目标点同步到robot_9失败: {e}")

    def shutdown(self):
        """关闭函数"""
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())  
        rospy.sleep(1)

    def get_robot_dimensions(self):
        """Returns the robot dimensions as [length1, length2, width]."""
        return [self.length1, self.length2, self.width]


    
    
    # ==================== 第1个奖励函数：计算行人接近软惩罚 ====================
    
    def _compute_reward_c(self, robot_position, robot_orientation, active_obstacles_mpi_states, 
                         d_m_warning=1.2, r_obstacle_penalty_factor=-0.2):
        """
        计算行人接近惩罚（collision avoidance soft penalty）
        
        功能说明：
        - 基于DRL-VO论文的r_c^t软惩罚机制
        - 当行人进入警告区域时，施加距离相关的惩罚
        - 累加所有在警告区域内的行人的惩罚值
        
        参数：
            robot_position (tuple): 机器人当前位置 (x, y)
            robot_orientation (tuple): 机器人当前朝向的cos和sin值 (cos_theta, sin_theta)
            active_obstacles_mpi_states (dict): MPI接收的动态障碍物状态字典
            d_m_warning (float): 警告区域阈值（米，中心到中心距离）
            r_obstacle_penalty_factor (float): 障碍物惩罚因子（负值）
            
        返回：
            float: 行人接近软惩罚总和（通常为负值或0）
        """
        # print(f"---------------------------active_obstacles_mpi_states: {active_obstacles_mpi_states}")
        x, y = robot_position
        cos_theta, sin_theta = robot_orientation
        
        # 累加所有行人的软惩罚
        total_r_c_soft_penalty = 0.0
        
        if not hasattr(self, 'dynamic_obstacles_mpi_states'):
            return total_r_c_soft_penalty
        
        # 遍历所有动态障碍物
        for robot_id in range(self.total_dynamic):  
            if robot_id not in active_obstacles_mpi_states:
                continue
                
            obstacle_state = active_obstacles_mpi_states[robot_id]
            
            # 跳过不活动或数据过期的障碍物
            if not obstacle_state.get('active', False):
                continue
            if (time.time() - obstacle_state.get('last_update', 0) >= 5.0):
                continue
            
            # 获取行人位置
            px_i, py_i, _ = obstacle_state['position']
            
            # 跳过无效数据
            if px_i == 0.0 and py_i == 0.0:
                continue
            
            # 计算相对位置（世界坐标系）
            dx = px_i - x
            dy = py_i - y
            
            # 计算中心到中心的距离
            dist_center_to_center = math.sqrt(dx**2 + dy**2)
            
            # 如果在警告区域内，则计算并累加惩罚
            if dist_center_to_center <= d_m_warning:
                ped_soft_penalty = r_obstacle_penalty_factor * (d_m_warning - dist_center_to_center)
                total_r_c_soft_penalty += ped_soft_penalty
        
        return total_r_c_soft_penalty
    


    # ============== 修改后的step函数（添加超时检测）==============

    def step(self):

        step_start_time = time.time()
        terminate = False
        self.stop_counter = 0
        reset = 0

        # 🔥 新增：初始化碰撞类型标志（用于logger统计）
        self.last_static_collision = False   # 是否与静态障碍物碰撞
        self.last_dynamic_collision = False  # 是否与动态障碍物碰撞
        
        
        # ============= 获取机器人状态（提前获取用于lidar匹配） =============
        [x, y, theta] = self.GetSelfStateGT()  # 真实位姿

        # ============= 获取激光雷达观测（540维） =============
        lidar_raw = self.GetNoisyLaserObservation()  # 获取带噪声的激光雷达数据

        laser_min = np.amin(lidar_raw)
        lidar_raw = np.reshape(lidar_raw, (540))

        # ============= 获取动态障碍物速度匹配结果 =============
        # 新增：通过ground truth位置匹配lidar点，区分动态/静态障碍物
        matched_velocities, is_dynamic = self.match_lidar_with_dynamic_obstacles(
            lidar_raw, [x, y, theta], self.dynamic_obstacles_mpi_states
        )

        # ============= 处理激光雷达数据（支持两种Pool策略）=============
        # 新格式 [90/180, 10]: [cos_α, sin_α, distance, vx, vy, is_dynamic, L1, L2, W, ttc_risk]
        # 方案A (use_dual_point_pool=False): [90, 10] 最近点策略
        # 方案B (use_dual_point_pool=True):  [180, 10] 双点策略（前90最近点，后90最近动态点）
        # ttc_risk: Time-To-Collision风险，考虑接近速度方向

        if self.use_dual_point_pool:
            # ============= 方案B：双点策略 [180, 10] =============
            pool_state = np.zeros((180, 10))

            for i in range(90):
                # 计算角度（前90和后90共享相同的角度）
                cos_alpha = np.cos(i * np.pi / 45.0 - np.pi / 2 + np.pi / 90)
                sin_alpha = np.sin(i * np.pi / 45.0 - np.pi / 2 + np.pi / 90)

                group_start = 6 * i
                group_end = group_start + 6
                group_distances = lidar_raw[group_start:group_end]
                group_is_dynamic = is_dynamic[group_start:group_end]

                # --- 前90个点：最近点信息 ---
                min_local_idx = np.argmin(group_distances)
                min_idx = group_start + min_local_idx
                nearest_dist = group_distances[min_local_idx]

                pool_state[i, 0] = cos_alpha
                pool_state[i, 1] = sin_alpha
                pool_state[i, 2] = nearest_dist

                # 附加速度和is_dynamic标志
                if is_dynamic[min_idx]:
                    vx = matched_velocities[min_idx, 0]
                    vy = matched_velocities[min_idx, 1]
                    pool_state[i, 3] = vx
                    pool_state[i, 4] = vy
                    pool_state[i, 5] = 1.0  # is_dynamic = True

                    # 计算TTC风险: v_approach = -(vx*cos + vy*sin)，正值表示接近
                    v_approach = -(vx * cos_alpha + vy * sin_alpha)
                    if v_approach > 0.05 and nearest_dist > 0.01:  # 正在接近且距离有效
                        ttc = nearest_dist / v_approach
                        ttc_risk = 1.0 / (ttc + 0.1)
                        ttc_risk = np.clip(ttc_risk, 0.0, 10.0)  # 截断防止数值爆炸
                    else:
                        ttc_risk = 0.0  # 不接近或静态，TTC风险为0
                else:
                    pool_state[i, 3] = 0.0
                    pool_state[i, 4] = 0.0
                    pool_state[i, 5] = 0.0  # is_dynamic = False
                    ttc_risk = 0.0  # 静态障碍物没有TTC风险

                pool_state[i, 6] = self.length1
                pool_state[i, 7] = self.length2
                pool_state[i, 8] = self.width
                pool_state[i, 9] = ttc_risk  # 新增: TTC风险特征

                # --- 后90个点：最近动态点信息（如无动态点则复制最近点）---
                if np.any(group_is_dynamic):
                    # 找到该组中最近的动态点
                    dynamic_distances = np.where(group_is_dynamic, group_distances, np.inf)
                    dynamic_local_idx = np.argmin(dynamic_distances)
                    dynamic_idx = group_start + dynamic_local_idx
                    dynamic_dist = dynamic_distances[dynamic_local_idx]

                    dyn_vx = matched_velocities[dynamic_idx, 0]
                    dyn_vy = matched_velocities[dynamic_idx, 1]

                    pool_state[90 + i, 0] = cos_alpha
                    pool_state[90 + i, 1] = sin_alpha
                    pool_state[90 + i, 2] = dynamic_dist
                    pool_state[90 + i, 3] = dyn_vx
                    pool_state[90 + i, 4] = dyn_vy
                    pool_state[90 + i, 5] = 1.0  # is_dynamic = True
                    pool_state[90 + i, 6] = self.length1
                    pool_state[90 + i, 7] = self.length2
                    pool_state[90 + i, 8] = self.width

                    # 计算动态点的TTC风险
                    v_approach_dyn = -(dyn_vx * cos_alpha + dyn_vy * sin_alpha)
                    if v_approach_dyn > 0.05 and dynamic_dist > 0.01:
                        ttc_dyn = dynamic_dist / v_approach_dyn
                        ttc_risk_dyn = 1.0 / (ttc_dyn + 0.1)
                        ttc_risk_dyn = np.clip(ttc_risk_dyn, 0.0, 10.0)
                    else:
                        ttc_risk_dyn = 0.0
                    pool_state[90 + i, 9] = ttc_risk_dyn
                else:
                    # 无动态点时，复制最近点信息
                    pool_state[90 + i, :] = pool_state[i, :]

                # 碰撞检测（使用最近点）
                x_dis = cos_alpha * nearest_dist
                y_dis = sin_alpha * nearest_dist
                if (abs(x_dis) <= self.width and
                    y_dis <= self.length1 and
                    y_dis >= -self.length2):
                    self.stop_counter += 1.0
                    if is_dynamic[min_idx]:
                        self.last_dynamic_collision = True
                    else:
                        self.last_static_collision = True

            pool_state = np.reshape(pool_state, (1800,))  # 180*10=1800

        else:
            # ============= 方案A：最近点策略 [90, 10] =============
            pool_state = np.zeros((90, 10))

            for i in range(90):
                cos_alpha = np.cos(i * np.pi / 45.0 - np.pi / 2 + np.pi / 90)
                sin_alpha = np.sin(i * np.pi / 45.0 - np.pi / 2 + np.pi / 90)

                pool_state[i, 0] = cos_alpha
                pool_state[i, 1] = sin_alpha

                # 找到该组中的最近点
                group_start = 6 * i
                group_end = group_start + 6
                group_distances = lidar_raw[group_start:group_end]
                min_local_idx = np.argmin(group_distances)
                min_idx = group_start + min_local_idx
                dis = group_distances[min_local_idx]

                x_dis = cos_alpha * dis
                y_dis = sin_alpha * dis

                pool_state[i, 2] = dis

                # 附加速度和is_dynamic标志
                if is_dynamic[min_idx]:
                    vx = matched_velocities[min_idx, 0]
                    vy = matched_velocities[min_idx, 1]
                    pool_state[i, 3] = vx
                    pool_state[i, 4] = vy
                    pool_state[i, 5] = 1.0  # is_dynamic = True

                    # 计算TTC风险
                    v_approach = -(vx * cos_alpha + vy * sin_alpha)
                    if v_approach > 0.05 and dis > 0.01:
                        ttc = dis / v_approach
                        ttc_risk = 1.0 / (ttc + 0.1)
                        ttc_risk = np.clip(ttc_risk, 0.0, 10.0)
                    else:
                        ttc_risk = 0.0
                else:
                    pool_state[i, 3] = 0.0
                    pool_state[i, 4] = 0.0
                    pool_state[i, 5] = 0.0  # is_dynamic = False
                    ttc_risk = 0.0

                pool_state[i, 6] = self.length1
                pool_state[i, 7] = self.length2
                pool_state[i, 8] = self.width
                pool_state[i, 9] = ttc_risk  # 新增: TTC风险特征

                #! ============= 适应尺寸的碰撞检测 =============
                if (abs(x_dis) <= self.width and
                    y_dis <= self.length1 and
                    y_dis >= -self.length2):
                    self.stop_counter += 1.0
                    if is_dynamic[min_idx]:
                        self.last_dynamic_collision = True
                    else:
                        self.last_static_collision = True

            pool_state = np.reshape(pool_state, (900,))  # 90*10=900

        
        # ============= 计算目标相对位置 =============
        self.pre_distance = copy.deepcopy(self.distance)
        
        abs_x = (self.target_point[0] - x) * self.ratio
        abs_y = (self.target_point[1] - y) * self.ratio
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        trans_matrix = np.matrix(
            [[np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]]
        )
        rela = np.matmul(trans_matrix, np.array([[abs_x], [abs_y]]))
        rela_x = rela[0, 0]
        rela_y = rela[1, 0]
        rela_distance = np.sqrt(rela_x**2 + rela_y**2)
        
        [v, w] = self.GetSelfSpeedGT()
        

        # ============= MPI广播（使用封装方法简化代码） =============
        self._broadcast_mpi_states(x, y, theta, v, w)

        self.distance = rela_distance
        
        # ============= 构造完整状态向量 =============
        rela_angle = np.arctan2(rela_y, rela_x)
        target_pose = [rela_distance, rela_angle]
        cur_act = [v, w]
        target = target_pose
        
        state = np.concatenate(
            [
                pool_state,
                target_pose,
                cur_act,
                [self.max_action[0], self.max_action[1],
                self.max_acc[0], self.max_acc[1]],
            ],
            axis=0,
        )
        
        # 计算或获取机器人有效半径
        if not hasattr(self, '_cached_robot_radius'):
            self._cached_robot_radius = math.sqrt(
                (self.width)**2 + (self.length1 + self.length2)**2
            ) / 2

        PED_EFFECTIVE_RADIUS = self.human_radius if hasattr(self, 'human_radius') else 0.3

        # ============= MPI动态障碍物数据接收（使用封装方法简化代码） =============
        self._receive_mpi_states()

        # ============= 行人硬碰撞检测（使用封装方法简化代码）=============
        ped_collision_detected = self._check_pedestrian_collision(x, y, cos_theta, sin_theta)
        if ped_collision_detected:
            self.stop_counter += 1.0
            self.last_dynamic_collision = True  # 🔥 标记动态碰撞
        

        # ============= 奖励配置参数 =============
        D_M_WARNING = 1.5
        R_OBSTACLE_PENALTY_FACTOR = -0.5
        
        # 计算行人接近软惩罚 reward_c
        reward_c = self._compute_reward_c(
            robot_position=(x, y),
            robot_orientation=(cos_theta, sin_theta),
            active_obstacles_mpi_states=getattr(self, 'dynamic_obstacles_mpi_states', {}),
            d_m_warning=D_M_WARNING,
            r_obstacle_penalty_factor=R_OBSTACLE_PENALTY_FACTOR,
        )
        
        
        # 4. 计算基础距离奖励 reward_g
        reward_g = 2 * (self.pre_distance - self.distance)
        
        
        # ============= 计算总奖励 =============
        reward = reward_g + reward_c  #+ reward_d       
        
        # ============= 检查卡住状态 =============
        if self.stalled:
            self.stop_counter += 1.0
        
        # ============= 终止条件判断 =============
        if self.stop_counter >= 1.0:
            # 碰撞终止
            reward = -10.0  # 碰撞惩罚-10
            terminate = True
            reset = 0
            self.crash_stop = True
            self.stop_counter = 0
            self.stalled = False
            # print(f"❌ Episode终止：碰撞！Steps={self.current_episode_steps}, reward={reward:.2f}")
        elif self.distance < 0.2 and not self.stalled:
            # 成功到达目标
            reward = 10.0  
            terminate = True
            reset = 1
            self.stop_counter = 0
            self.stalled = False
            # print(f"✅ Episode成功！Steps={self.current_episode_steps}, reward={reward:.2f}")
        
        return state, reward, terminate, reset, self.distance, [x, y, theta]



    def GenerateTargetPoint(self, suc_rate):
        """生成目标点"""
        local_window = self.map_size[0]
        [xx, yy, theta] = self.GetSelfStateGT()
        xx = xx - self.map_center[self.env, 0]
        yy = yy - self.map_center[self.env, 1]
        
        max_attempts = 100
        attempt_count = 0
        
        x = random.uniform(
            max(-(self.map_size[0]/2 - self.target_size), xx - local_window), 
            min((self.map_size[0]/2 - self.target_size), xx + local_window)
        )
        y = random.uniform(
            max(-(self.map_size[1]/2 - self.target_size), yy - local_window), 
            min((self.map_size[1]/2 - self.target_size), yy + local_window)
        )
        
        while (not self.targetPointCheck(x, y) and 
            not rospy.is_shutdown() and 
            attempt_count < max_attempts):
            
            attempt_count += 1
            
            x = random.uniform(
                max(-(self.map_size[0]/2 - self.target_size), xx - local_window), 
                min((self.map_size[0]/2 - self.target_size), xx + local_window)
            )
            y = random.uniform(
                max(-(self.map_size[1]/2 - self.target_size), yy - local_window), 
                min((self.map_size[1]/2 - self.target_size), yy + local_window)
            )
        
        if attempt_count >= max_attempts and not self.targetPointCheck(x, y):
            print(f"WARNING: 目标点生成达到最大尝试次数({max_attempts})，使用备用策略")
            
            backup_attempts = 50
            for i in range(backup_attempts):
                x = random.uniform(
                    -(self.map_size[0]/2 - self.target_size), 
                    (self.map_size[0]/2 - self.target_size)
                )
                y = random.uniform(
                    -(self.map_size[1]/2 - self.target_size), 
                    (self.map_size[1]/2 - self.target_size)
                )
                
                if self.targetPointCheck(x, y):
                    break
            else:
                print(f"  备用策略也失败，强制使用最后位置: [{x:.3f}, {y:.3f}]")
        
        self.target_point = [x + self.map_center[self.env, 0], y + self.map_center[self.env, 1]]
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

        # ! publish goal
        self.publish_target_point(self.target_point)  # 添加这一行

        # === MPI同步目标点（rank 3）===
        # 使用Tag 300发送目标点位置给目标点发布进程
        if hasattr(self, 'mpi_comm') and self.mpi_comm is not None:
            try:
                # 准备目标点消息
                target_msg = {
                    "command": "target_update",
                    "target_position": self.target_point,  # [x, y]
                    "timestamp": time.time()  # 添加时间戳用于调试
                }
             
                # 注意：rank 0=主控机器人, rank 1=robot_1, rank 2=robot_2, rank 3=robot_9
                self.mpi_comm.send(target_msg, dest=3, tag=300)
                # rospy.logdebug(f"目标点已通过MPI发送给robot_45: {self.target_point}")
            except Exception as e:
                rospy.logwarn(f"MPI发送目标点到robot_9失败: {e}")
                import traceback
                traceback.print_exc()

    def GetLocalTarget(self):
        """获取目标点在机器人坐标系中的位置"""
        [x, y, theta] = self.GetSelfStateGT()
        [target_x, target_y] = self.target_point
        
        local_x = (target_x - x) * np.cos(theta) + (target_y - y) * np.sin(theta)
        local_y = -(target_x - x) * np.sin(theta) + (target_y - y) * np.cos(theta)
        
        return [local_x, local_y]

    def TargetPointCheck(self):
        """检查当前目标点是否有效"""
        target_x = self.target_point[0]
        target_y = self.target_point[1]
        pass_flag = True
        
        x_pixel = int(target_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(target_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.robot_range * np.amax(self.R2P))
        
        for x in range(
            np.amax([0, x_pixel - window_size]),
            np.amin([self.map_pixel[0] - 1, x_pixel + window_size]),
        ):
            for y in range(
                np.amax([0, y_pixel - window_size]),
                np.amin([self.map_pixel[1] - 1, y_pixel + window_size]),
            ):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        
        return pass_flag

    def Control(self, action):
        """执行控制动作"""
        # �� 添加调试信息
        if self.robot_index == 0:  # 只为主控机器人打印调试信息
            # print(f"�� 主控机器人Control被调用: action={action}")
            pass
        
        [v, w] = self.GetSelfSpeed()
        
        # scaled_linear_action = (action[0] + 1.0) / 2.0 

        # 使用重新映射后的值来计算目标速度
        # target_linear_vel = scaled_linear_action * self.max_action[0]
        
        self.self_speed[0] = np.clip(
            action[0] * self.max_action[0],    
            # target_linear_vel,                
            v - self.max_acc[0] * self.control_period,         
            v + self.max_acc[0] * self.control_period,         
        )
        
        self.self_speed[1] = np.clip(
            action[1] * self.max_action[1],                    
            w - self.max_acc[1] * self.control_period,         
            w + self.max_acc[1] * self.control_period,         
        )
        
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]   
        move_cmd.linear.y = 0.                   
        move_cmd.linear.z = 0.                   
        move_cmd.angular.x = 0.                  
        move_cmd.angular.y = 0.                  
        move_cmd.angular.z = self.self_speed[1]  

        self.cmd_vel.publish(move_cmd)

    def PIDController(self):
        """PID控制器"""
        action_bound = self.max_action
        X = self.GetSelfState()  
        X_t = self.GetLocalTarget()  
        P = np.array([10.0, 1.0])  
        
        Ut = X_t * P

        if Ut[0] < -action_bound[0]:
            Ut[0] = -action_bound[0]
        elif Ut[0] > action_bound[0]:
            Ut[0] = action_bound[0]

        if Ut[1] < -action_bound[1]:
            Ut[1] = -action_bound[1]
        elif Ut[1] > action_bound[1]:
            Ut[1] = action_bound[1]
        
        Ut[0] = Ut[0] / self.max_action[0]
        Ut[1] = Ut[1] / self.max_action[1]

        return Ut

    
    def send_my_state_to_main_process(self):
        """
        �� 动态障碍物进程：将自己的状态发送给主进程
        """
        if (self.mpi_comm is not None and 
            self.mpi_rank != 0 and 
            hasattr(self, 'object_state') and 
            len(self.object_state) >= 4):
            
            try:
                # 构造状态消息
                state_msg = {
                    'robot_id': self.robot_index,
                    'position': [self.object_state[0], self.object_state[1], 0.0],
                    'velocity': [self.object_state[2], self.object_state[3]],
                    'timestamp': time.time()
                }
                
                # 非阻塞发送给主进程
                self.mpi_comm.isend(state_msg, dest=0, tag=200)
                
            except Exception as e:
                rospy.logwarn(f"MPI状态发送失败: {e}")
    

    def set_mpi_comm(self, mpi_comm, mpi_rank, mpi_size):
        """
        设置MPI通信参数（使用新的 MPIHandler）

        Args:
            mpi_comm: MPI通信器
            mpi_rank: MPI进程rank
            mpi_size: MPI进程总数
        """
        # 创建 MPIHandler 实例
        self.mpi_handler = MPIHandler(comm=mpi_comm, rank=mpi_rank, size=mpi_size)

        # 保留旧的属性以保持向后兼容（标记为废弃）
        self.mpi_comm = mpi_comm  # 废弃：请使用 self.mpi_handler
        self.mpi_rank = mpi_rank  # 废弃：请使用 self.mpi_handler.rank
        self.mpi_size = mpi_size  # 废弃：请使用 self.mpi_handler.size

        rospy.loginfo(f"MPI通信设置完成: rank {mpi_rank}/{mpi_size} (使用 MPIHandler)")

    def set_mpi_handler(self, mpi_handler):
        """
        直接设置 MPIHandler 实例（推荐的新方法）

        Args:
            mpi_handler: MPIHandler 实例
        """
        self.mpi_handler = mpi_handler

        # 同步旧属性以保持向后兼容
        self.mpi_comm = mpi_handler.mpi_comm if mpi_handler else None
        self.mpi_rank = mpi_handler.rank if mpi_handler else 0
        self.mpi_size = mpi_handler.size if mpi_handler else 1

        rospy.loginfo(f"MPIHandler 设置完成: rank {self.mpi_rank}/{self.mpi_size}")


    def _get_random_inside_position(self):
        """在地图内生成随机位置"""
        x = random.uniform(
            self.area_center_x - self.area_half,
            self.area_center_x + self.area_half
        )
        y = random.uniform(-self.area_half, self.area_half)
        return (x, y)

    def _get_safe_random_position(self):
        """生成一个安全的随机位置（保证不在障碍物内）"""
        x = random.uniform(
            self.area_center_x - self.area_half,
            self.area_center_x + self.area_half
        )
        y = random.uniform(-self.area_half, self.area_half)

        while not self.pedestrianPointCheck(x, y) and not rospy.is_shutdown():
            x = random.uniform(
                self.area_center_x - self.area_half,
                self.area_center_x + self.area_half
            )
            y = random.uniform(-self.area_half, self.area_half)

        return (x, y)

    # ========== 动态障碍物状态管理的新接口（使用 DynamicObstacleStateManager） ==========

    def Time_dynamic_obstacles(self):
        """
        获取动态障碍物的时序信息（新版本：移除时序维度）

        注意：time_steps 参数已废弃，保留仅为向后兼容

        Args:
            time_steps: (废弃) 时间步数，当前版本忽略此参数

        Returns:
            np.array: shape [N, 6]，N为当前激活的动态障碍物数量
                     特征: [相对px, 相对py, 相对vx, 相对vy, 距离, 半径]
        """


        # 使用新的 DynamicObstacleStateManager 获取当前帧障碍物
        return self.obstacle_manager.get_current_dynamic_obstacles(self)

    def get_robot_state_for_joint(self):
        """
        获取机器人状态用于joint state构建（委托给 DynamicObstacleStateManager）

        Returns:
            dict: 包含机器人的位置、速度、朝向等信息
        """
        return DynamicObstacleStateManager.get_robot_state_for_joint(self)

    def rotate_joint_state(self, robot_state, obstacle_state):
        """
        将障碍物状态转换到机器人自我中心坐标系（委托给 DynamicObstacleStateManager）

        Args:
            robot_state: 机器人状态字典
            obstacle_state: 障碍物状态字典

        Returns:
            np.array: shape [6]，转换后的障碍物特征
        """
        return DynamicObstacleStateManager.rotate_joint_state(robot_state, obstacle_state)

    # ========== MPI通信和碰撞检测的封装方法（简化step函数） ==========

    def _receive_mpi_states(self):
        """
        接收MPI动态障碍物状态（私有方法，在step中调用）

        功能：
            - 使用MPIHandler接收来自动态障碍物进程的状态更新
            - 安全地更新环境中的障碍物状态字典
            - 只有主控机器人才执行此操作
        """
        if not (hasattr(self, 'dynamic_obstacles_mpi_states') and
                hasattr(self, 'mpi_handler') and
                self.mpi_handler is not None):
            return

        if not self.is_main_robot:
            return

        # 使用 MPIHandler 接收动态障碍物状态更新
        # 🔥 修复：receive_dynamic_states()返回生成器，需要正确迭代处理
        for robot_id, position, velocity, active in self.mpi_handler.receive_dynamic_states():
            self.dynamic_obstacles_mpi_states[robot_id] = {
                'position': position,
                'velocity': velocity,
                'active': active,
                'last_update': time.time()
            }

    def _broadcast_mpi_states(self, x, y, theta, v, w):
        """
        广播MPI状态（私有方法，在step中调用）

        Args:
            x, y: 机器人位置
            theta: 机器人朝向
            v: 线速度
            w: 角速度

        功能：
            - 广播主控机器人状态给所有动态障碍物进程
            - 广播所有行人状态给所有动态障碍物进程
            - 只有主控机器人才执行此操作
        """
        if not (self.is_main_robot and
                hasattr(self, 'mpi_handler') and
                self.mpi_handler is not None):
            return

        # 1. 构造主控机器人自己的状态
        robot_state_msg = {
            'x': x,
            'y': y,
            'theta': theta,
            'v': v,
            'w': w,
            'timestamp': time.time()
        }

        # 2. 获取所有动态障碍物的状态字典
        # (由 mpi_handler.receive_dynamic_states() 填充到 self.dynamic_obstacles_mpi_states)
        all_obstacles_state_msg = getattr(self, 'dynamic_obstacles_mpi_states', {})

        # 3. 使用 MPIHandler 广播状态
        try:
            # 广播主控机器人状态 (TAG_ROBOT_STATE = 201)
            self.mpi_handler.broadcast_robot_state(robot_state_msg, total_dynamic=self.total_dynamic)

            # 广播所有行人（动态障碍物）状态 (TAG_PEDESTRIAN_STATES = 202)
            self.mpi_handler.broadcast_all_pedestrians(all_obstacles_state_msg, total_dynamic=self.total_dynamic)

            # 添加调试日志（每10步打印一次）
            if not hasattr(self, '_mpi_broadcast_counter'):
                self._mpi_broadcast_counter = 0

            self._mpi_broadcast_counter += 1
            if self._mpi_broadcast_counter % 50 == 0:  # 降低日志频率
                active_count = sum(1 for state in all_obstacles_state_msg.values() if state.get('active', False))
                # rospy.loginfo_throttle(5, f"MPI广播: 激活障碍物={active_count}/{len(all_obstacles_state_msg)}")

        except Exception as e:
            rospy.logwarn_throttle(10, f"MPI广播失败: {e}")

    def _check_pedestrian_collision(self, x, y, cos_theta, sin_theta):
        """
        检测行人碰撞（私有方法，在step中调用）

        Args:
            x, y: 机器人位置
            cos_theta, sin_theta: 机器人朝向的cos和sin值

        Returns:
            bool: 是否检测到碰撞
        """
        # 检查collision_detector是否已创建（延迟初始化）
        if not hasattr(self, 'collision_detector') or self.collision_detector is None:
            rospy.logwarn_once("CollisionDetector未初始化（需要先调用ResetWorld或reset），跳过碰撞检测")
            return False

        if not hasattr(self, 'dynamic_obstacles_mpi_states'):
            return False

        # 使用CollisionDetector检测碰撞
        collision_info = self.collision_detector.check_pedestrian_collision(
            robot_position=(x, y),
            robot_orientation=(cos_theta, sin_theta),
            obstacles_states=self.dynamic_obstacles_mpi_states,
            check_ids=list(range(1, self.total_dynamic + 1))  # 检查所有动态障碍物
        )

        return collision_info['detected']
