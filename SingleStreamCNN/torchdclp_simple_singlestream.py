# -*- coding: utf-8 -*-
"""
简化版SAC训练文件 - 支持方案A/B切换 + TDE时序编码器

方案A: 最近点策略，90个点，10维特征（含ttc_risk），state维度 = 908
方案B: 双点策略，180个点，10维特征（含ttc_risk），state维度 = 1808

新增特性（借鉴DARE论文）：
- ttc_risk: Time-To-Collision风险特征
- TDE: Temporal Dynamic Encoder 时序动态编码器（可选）
- 梯度解耦: TDE独立训练，不干扰RL

使用说明：
1. 设置 USE_DUAL_POINT_POOL = False 使用方案A (torchcore_true.py)
2. 设置 USE_DUAL_POINT_POOL = True 使用方案B (torchcore_true_dual.py)
3. 设置 USE_TDE = True 启用时序动态编码器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.robot_control.orca_controller import ORCAController
from utils.robot_control.run_mpi_dynamic_obstacle_control_ORCA import run_mpi_dynamic_obstacle_control
from utils.temporal_dynamic_encoder import TDEManager  # 新增: TDE模块
import gym
import time
import rospy
import os
import signal
import subprocess
import sys
from collections import deque
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import truncnorm
from geometry_msgs.msg import Pose2D, Twist

# ============= 🔥 方案配置开关 =============
USE_DUAL_POINT_POOL = True  # False: 方案A (90点), True: 方案B (180点)
USE_TDE = True  # 是否启用Temporal Dynamic Encoder (借鉴DARE)

# 根据配置导入对应的网络模块
# 新增ttc_risk特征，每点从9维变为10维
if USE_DUAL_POINT_POOL:
    import torchcore_true_dual as core
    from torchcore_true_dual import init_weights_xavier
    OBS_DIM = 180 * 10 + 8  # 1808 (原1628，新增ttc_risk)
    POOL_SIZE = 180 * 10    # 点云数据大小
    NUM_POINTS = 180
    print("📌 使用方案B: 双点策略 (180点 × 10维 + 8 = 1808)")
else:
    import torchcore_true as core
    from torchcore_true import init_weights_xavier
    OBS_DIM = 90 * 10 + 8   # 908 (原818，新增ttc_risk)
    POOL_SIZE = 90 * 10     # 点云数据大小
    NUM_POINTS = 90
    print("📌 使用方案A: 最近点策略 (90点 × 10维 + 8 = 908)")

if USE_TDE:
    print("📌 启用TDE: Temporal Dynamic Encoder (借鉴DARE)")
else:
    print("📌 禁用TDE: 不使用时序动态编码器")

from stage_obs_dyn_curlearning_grid36_fixed_size import StageWorld
from training_logger import TrainingLogger

#! ============= 🔥 MPI多进程支持 🔥 =============
from mpi4py import MPI

# ============= 🔥 MPI通信标签定义 🔥 =============
TAG_CURRICULUM_UPDATE = 100
TAG_OBSTACLE_COMMAND = 101
TAG_SHUTDOWN = 999

# ============= 🔥 MPI进程分配常量 🔥 =============
NUM_TOTAL_PROCESSES = 4
NUM_DYNAMIC_OBSTACLES = 2
MAIN_ROBOT_RANK = 0
DYNAMIC_OBSTACLE_START_RANK = 1


class ReplayBuffer:
    """
    简化版经验回放缓冲区 - 移除time_dynamic存储
    """

    def __init__(self, obs_dim, act_dim, size):
        """
        初始化经验缓冲区（简化版本：移除time_dynamic）
        """
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        """
        存储一条经验（简化版本）
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=128, start=0):
        """采样训练批次"""
        idxs = np.random.randint(int(start), self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


class StageWorldLogger(StageWorld):
    """扩展StageWorld类，添加日志功能和MPI支持"""

    def __init__(self, beam_num, logger=None, index=0, num_env=1):
        super().__init__(beam_num, index=index, num_env=num_env)
        self.logger = logger
        self.step_context = "training"
        # 设置Pool策略配置
        self.use_dual_point_pool = USE_DUAL_POINT_POOL

    def set_step_context(self, context):
        """设置step调用的上下文"""
        self.step_context = context

    def step(self):
        """重写step函数，添加上下文感知的日志"""
        state, reward, terminate, reset, distance, robot_pose = super().step()

        if terminate and reset == 0:
            if self.logger and self.step_context == "initialization":
                self.logger.log_crash("initialization")

        return state, reward, terminate, reset, distance, robot_pose


def sac(
    actor_critic=core.MLPActorCritic,
    seed=5,
    steps_per_epoch=5000,
    epochs=10000,
    replay_size=int(2e6),
    gamma=0.99,
    polyak=0.995,
    lr1=1e-4,
    lr2=1e-4,
    alpha=0.01,
    batch_size=100,
    start_epoch=100,
    max_ep_len=400,
    MAX_EPISODE=10000,
    device='cpu',
    mpi_rank=0,
    mpi_size=1,
    robot_index=0,
    is_main_robot=True,
    mpi_comm=None,
):
    """
    简化版SAC算法实现 - 移除hit_planner和time_dynamic
    """

    print(f"🔥 MPI进程 {mpi_rank}/{mpi_size}: {'主控机器人' if is_main_robot else f'动态障碍物{robot_index}'}开始训练")

    # 只有主进程执行完整训练
    if not is_main_robot:
        if robot_index == 9:
            print(f"🎯 进程 {mpi_rank}: 目标点发布控制模式")
        else:
            print(f"🤖 进程 {mpi_rank}: 动态障碍物控制模式")
            run_mpi_dynamic_obstacle_control(robot_index, mpi_rank, mpi_comm)
        return

    # ============= 基础参数设置 =============
    sac_id = 1
    obs_dim = OBS_DIM  # 使用配置的观测维度
    act_dim = 2

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 主进程 {mpi_rank} 使用设备: {device}")
    print(f"📐 观测维度: {obs_dim}")

    # ============= 构建神经网络 =============
    main_model = actor_critic(obs_dim, act_dim).to(device)
    target_model = actor_critic(obs_dim, act_dim).to(device)

    main_model.apply(init_weights_xavier)
    print("网络参数已应用Xavier初始化")

    target_model.load_state_dict(main_model.state_dict())

    for param in target_model.parameters():
        param.requires_grad = False

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # ============= TDE初始化（借鉴DARE论文）=============
    tde_manager = None
    if USE_TDE:
        tde_manager = TDEManager(
            device=device,
            input_channels=10,  # 每点10维特征
            num_dynamic_points=90,  # 后90个点是动态点
            embedding_dim=64,
            action_dim=act_dim,
            lr=3e-4,
            tau=0.005
        )
        print(f"📌 TDE初始化完成: embedding_dim=64, 动态点数=90")

    # ============= 网络参数统计 =============
    total_params = core.count_vars(main_model)
    policy_params = core.count_vars(main_model.policy)
    q1_params = core.count_vars(main_model.q1)
    q2_params = core.count_vars(main_model.q2)
    cnn_params = core.count_vars(main_model.cnn_dense)

    print(f'\n========== 网络参数统计 ==========')
    print(f'总参数数量: {total_params}')
    print(f'策略网络总参数: {policy_params}')
    print(f'CNN特征提取器参数: {cnn_params}')
    print(f'Q1网络: {q1_params}')
    print(f'Q2网络: {q2_params}')
    print(f'=====================================================\n')

    # ============= 优化器设置（简化版：移除hit_planner参数）=============
    pi_optimizer = optim.Adam(main_model.policy.parameters(), lr=lr2)

    q_optimizer = optim.Adam([
        *main_model.q1.parameters(),
        *main_model.q2.parameters(),
        *main_model.cnn_dense.parameters(),  # CNN特征提取器参数
    ], lr=lr1)

    l2_reg = 0.001

    def get_action(o, deterministic=False):
        """获取动作（简化版：移除time_dynamic）"""
        with torch.no_grad():
            o_tensor = torch.FloatTensor(o.reshape(1, -1)).to(device)
            mu, pi, logp_pi, *_ = main_model(o_tensor)

            if deterministic:
                return mu.cpu().numpy()[0]
            else:
                return pi.cpu().numpy()[0]

    def update_networks(batch):
        """更新神经网络（简化版：移除time_dynamic）"""
        obs1 = torch.FloatTensor(batch['obs1']).to(device)
        obs2 = torch.FloatTensor(batch['obs2']).to(device)
        acts = torch.FloatTensor(batch['acts']).to(device)
        rews = torch.FloatTensor(batch['rews']).to(device)
        done = torch.FloatTensor(batch['done']).to(device)

        # ============= 更新Q网络 =============
        with torch.no_grad():
            _, pi_next, logp_pi_next, q1_pi_targ, q2_pi_targ, *_ = target_model(obs2)
            min_q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rews + gamma * (1 - done) * (min_q_pi_targ - alpha * logp_pi_next)

        # Q网络更新
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = main_model(obs1, acts)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        q_loss = q1_loss + q2_loss

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # ============= 更新策略网络 =============
        for param in main_model.q1.parameters():
            param.requires_grad = False
        for param in main_model.q2.parameters():
            param.requires_grad = False
        for param in main_model.cnn_dense.parameters():
            param.requires_grad = False

        mu, pi, logp_pi, q1_pi, q2_pi = main_model(obs1)
        min_q_pi = torch.min(q1_pi, q2_pi)
        l2_penalty = sum(param.pow(2.0).sum() for param in main_model.policy.parameters())

        pi_loss = torch.mean(alpha * logp_pi - min_q_pi) + l2_reg * l2_penalty

        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        for param in main_model.q1.parameters():
            param.requires_grad = True
        for param in main_model.q2.parameters():
            param.requires_grad = True
        for param in main_model.cnn_dense.parameters():
            param.requires_grad = True

        # ============= 软更新目标网络 =============
        with torch.no_grad():
            for param_main, param_target in zip(main_model.parameters(), target_model.parameters()):
                param_target.data.mul_(polyak)
                param_target.data.add_((1 - polyak) * param_main.data)

        # ============= TDE更新（梯度解耦，独立训练）=============
        tde_loss_info = {}
        if USE_TDE and tde_manager is not None:
            tde_loss_info = tde_manager.update(
                obs=obs1,
                obs_next=obs2,
                actions=acts,
                rewards=rews,
                pool_size=POOL_SIZE
            )

        return pi_loss.item(), q_loss.item(), tde_loss_info

    # ============= TensorBoard设置 =============
    log_dir = f'./logssac{sac_id}'
    summary_writer = SummaryWriter(log_dir)

    # ============= 训练初始化 =============
    episode = 0
    T = 0
    epi_thr = 0

    try:
        test_result_plot = np.load('test_result_plot1.npy')
    except FileNotFoundError:
        test_result_plot = np.zeros((5, 5, 50, 101, 5))
        print("创建新的测试结果存储数组")

    test_time = 0

    # ============= 多实验循环 =============
    for hyper_exp in range(1, 5):
        print(f"\n========== 🔥 简化版训练实验 {hyper_exp} 开始 ==========")
        print(f"📌 Pool策略: {'双点(180)' if USE_DUAL_POINT_POOL else '最近点(90)'}")

        logger = TrainingLogger(experiment_id=hyper_exp)

        env = StageWorldLogger(540, logger=logger, index=mpi_rank, num_env=mpi_size)
        env.set_mpi_comm(mpi_comm, mpi_rank, mpi_size)

        rate = rospy.Rate(10)

        goal_reach = 0
        suc_record = np.zeros((5, 50))
        suc_record1 = np.zeros((5, 50))
        suc_record2 = np.zeros((5, 50))
        suc_pointer = np.zeros(5)
        mean_rate = np.zeros(5)
        env_list = np.zeros(5)
        p = np.zeros(9)
        p[0] = 1.0
        mean_rate[0] = 0.0

        seed = hyper_exp
        torch.manual_seed(seed)
        np.random.seed(seed)

        main_model = actor_critic(obs_dim, act_dim).to(device)
        target_model = actor_critic(obs_dim, act_dim).to(device)
        main_model.apply(init_weights_xavier)
        target_model.load_state_dict(main_model.state_dict())

        for param in target_model.parameters():
            param.requires_grad = False

        pi_optimizer = optim.Adam(main_model.policy.parameters(), lr=lr2)
        q_optimizer = optim.Adam([
            *main_model.q1.parameters(),
            *main_model.q2.parameters(),
            *main_model.cnn_dense.parameters(),
        ], lr=lr1)

        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        episode = 0
        T = 0
        epi_thr = 0
        goal_reach = 0
        test_time = 0
        new_env = 0
        b_test = True
        length_index = 0

        # ============= 主训练循环 =============
        while test_time < 101:

            if new_env == 0:
                env_no = 0
            else:
                env_no = new_env

            length_index = 0
            length1 = 0.1+0.015
            length2 = 0.1+0.015
            width = 0.1+0.015

            env_no = int(env_list[length_index])

            logger.set_phase("TRAINING",
                            env=int(env_list[length_index]),
                            robot_size=[length1, length2, width])

            T_step = 0
            goal_reach = 0
            env.set_step_context("initialization")

            if goal_reach == 1 and b_test:
                robot_size = env.Reset(env_no)
            else:
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                b_test = True

            env.GenerateTargetPoint(mean_rate[length_index])
            o, r, d, goal_reach, r2gd, robot_pose = env.step()
            rate.sleep()

            try_time = 0
            while r2gd < 0.3 and try_time < 100:
                try_time = try_time + 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()

            try_time = 0
            while d and try_time < 100:
                try_time = try_time + 1
                robot_size = env.ResetWorld(env_no, length1, length2, width)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()

            try_time = 0
            while r2gd < 0.3 and try_time < 1000:
                try_time = try_time + 1
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d, goal_reach, r2gd, robot_pose = env.step()
                rate.sleep()

            max_ep_len = int(40 / robot_size * 5)

            robot_dims = f"({length1:.2f},{length2:.2f},{width:.2f})"
            logger.write_log(
                f"🚀 [训练] Ep: {episode:4d} | "
                f"尺寸组:{length_index} (环境Lvl:{env_no}) | "
                f"机器人尺寸:{robot_dims} | "
                f"初始距离:{r2gd:.1f}m | 最大步数:{max_ep_len}"
            )

            env.set_step_context("training")

            reset = False
            return_epoch = 0
            total_vel = 0
            ep_len = 0
            d = False
            last_d = 0

            # ============= Episode主循环 =============
            while not reset:
                if episode > start_epoch:
                    a = get_action(o, deterministic=False)
                else:
                    a = env.PIDController()
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()

                o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()

                return_epoch = return_epoch + r
                total_vel = total_vel + a[0]

                # 简化版：不存储time_dynamic
                replay_buffer.store(o, a, r, o2, d)
                ep_len += 1

                o = o2
                last_d = d

                if d:
                    if episode > start_epoch:
                        suc_record[length_index, int(suc_pointer[length_index])] = goal_reach

                        if env.stop_counter < 1.0:
                            suc_record1[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index, int(suc_pointer[length_index])] = 0
                        else:
                            suc_record1[length_index, int(suc_pointer[length_index])] = 0.0
                            suc_record2[length_index, int(suc_pointer[length_index])] = 0

                        suc_pointer[length_index] = (suc_pointer[length_index] + 1) % 50

                    logger.log_episode_end(
                        episode=episode,
                        reward=return_epoch,
                        steps=ep_len,
                        success=bool(goal_reach),
                        crash=(not bool(goal_reach)),
                        timeout=False,
                        update_training_steps=(episode > start_epoch),
                        static_collision=getattr(env, 'last_static_collision', False),
                        dynamic_collision=getattr(env, 'last_dynamic_collision', False)
                    )
                    reset = True
                else:
                    if ep_len >= max_ep_len:
                        if episode > start_epoch:
                            suc_record[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record1[length_index, int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index, int(suc_pointer[length_index])] = 1.0
                            suc_pointer[length_index] = (suc_pointer[length_index] + 1) % 50

                        logger.log_episode_end(
                            episode=episode,
                            reward=return_epoch,
                            steps=ep_len,
                            success=bool(goal_reach),
                            crash=False,
                            timeout=True,
                            update_training_steps=(episode > start_epoch),
                            static_collision=getattr(env, 'last_static_collision', False),
                            dynamic_collision=getattr(env, 'last_dynamic_collision', False)
                        )
                        reset = True

                # ============= 网络训练 =============
                if episode > start_epoch:
                    if goal_reach == 1 or env.crash_stop or (ep_len >= max_ep_len):
                        if ep_len == 0:
                            ep_len = 1

                        reset = True
                        average_vel = total_vel / ep_len

                        for j in range(ep_len):
                            T = T + 1

                            start = np.minimum(
                                replay_buffer.size * (1.0 - (0.996 ** (j * 1.0 / ep_len * 1000.0))),
                                np.maximum(replay_buffer.size - 10000, 0),
                            )

                            batch = replay_buffer.sample_batch(batch_size, start=start)
                            pi_loss, q_loss, tde_loss_info = update_networks(batch)

                            if T % 100 == 0:
                                summary_writer.add_scalar('Loss/Policy', pi_loss, T)
                                summary_writer.add_scalar('Loss/Q_Network', q_loss, T)
                                # TDE损失记录
                                if USE_TDE and tde_loss_info:
                                    summary_writer.add_scalar('Loss/TDE_Total', tde_loss_info.get('tde_loss', 0), T)
                                    summary_writer.add_scalar('Loss/TDE_Pred', tde_loss_info.get('tde_pred_loss', 0), T)
                                    summary_writer.add_scalar('Loss/TDE_Reward', tde_loss_info.get('tde_reward_loss', 0), T)

                            # ============= 定期性能测试 =============
                            if T % 10000 == 0:
                                logger.set_phase("TESTING", test_round=T//10000)
                                logger.start_test_round(T//10000)

                                env.set_step_context("testing")

                                group_results = {}

                                for shape_no in range(1):
                                    logger.log_test_group_start(shape_no, int(env_list[shape_no]))

                                    group_rewards = []
                                    group_success = []
                                    group_crashes = []

                                    for k in range(50):
                                        total_vel_test = 0
                                        return_epoch_test = 0
                                        ep_len_test = 0

                                        rospy.sleep(1.0)

                                        velcity = env.set_robot_pose_test(k, int(env_list[shape_no]), shape_no)
                                        env.GenerateTargetPoint_test(k, int(env_list[shape_no]), shape_no)
                                        max_ep_len = int(40 / velcity * 5)

                                        o, r, d, goal_reach, r2gd, robot_pose = env.step()

                                        for i in range(1000):
                                            a = get_action(o, deterministic=True)

                                            env.Control(a)
                                            rate.sleep()
                                            env.Control(a)
                                            rate.sleep()

                                            o2, r, d, goal_reach, r2gd, robot_pose2 = env.step()
                                            return_epoch_test = return_epoch_test + r
                                            total_vel_test = total_vel_test + a[0]
                                            ep_len_test += 1
                                            o = o2

                                            if d or (ep_len_test >= max_ep_len):
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 0] = return_epoch_test
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 1] = goal_reach
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 2] = (ep_len_test * 1.0 / max_ep_len)
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 3] = (1.0 * goal_reach - ep_len_test * 2.0 / max_ep_len)
                                                test_result_plot[hyper_exp, shape_no, k, test_time, 4] = env.crash_stop

                                                group_rewards.append(return_epoch_test)
                                                group_success.append(goal_reach)
                                                group_crashes.append(env.crash_stop)

                                                break

                                    group_result = {
                                        'success_rate': np.mean(group_success),
                                        'avg_reward': np.mean(group_rewards),
                                        'collision_rate': np.mean(group_crashes),
                                        'test_count': len(group_success)
                                    }
                                    group_results[shape_no] = group_result

                                    logger.log_test_group_end(shape_no, group_result)

                                    mean_rate[shape_no] = np.mean(
                                        test_result_plot[hyper_exp, shape_no, :, test_time, 1]
                                    )

                                    np.save(f'test_result_plot{sac_id}.npy', test_result_plot)

                                logger.end_test_round(env_list)
                                test_time = test_time + 1

                                logger.plot_learning_curves()
                                logger.plot_test_results_heatmap()
                                logger.save_training_data()

                                if (test_time) % 1 == 0:
                                    model_dir = './saved_models'
                                    os.makedirs(model_dir, exist_ok=True)

                                    pool_type = 'dual180' if USE_DUAL_POINT_POOL else 'nearest90'
                                    save_path = os.path.join(
                                        model_dir,
                                        f'simple_{pool_type}_{sac_id}lambda{hyper_exp}_{test_time}.pth'
                                    )

                                    save_dict = {
                                        'main_model_state_dict': main_model.state_dict(),
                                        'target_model_state_dict': target_model.state_dict(),
                                        'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                                        'q_optimizer_state_dict': q_optimizer.state_dict(),
                                        'test_time': test_time,
                                        'T': T,
                                        'episode': episode,
                                        'pool_type': pool_type,
                                        'obs_dim': obs_dim,
                                    }
                                    # 保存TDE状态
                                    if USE_TDE and tde_manager is not None:
                                        save_dict['tde_state'] = tde_manager.tde.state_dict()
                                        save_dict['tde_target_state'] = tde_manager.tde_target.state_dict()
                                        save_dict['tde_optimizer_state'] = tde_manager.optimizer.state_dict()
                                        save_dict['tde_update_count'] = tde_manager.update_count

                                    torch.save(save_dict, save_path)

                                    logger.write_log(f"💾 模型已保存: {save_path}")
                                    rospy.sleep(1.5)

                                logger.set_phase("TRAINING")
                                env.set_step_context("training")
                                b_test = False

            episode = episode + 1
            epi_thr = epi_thr + 1

        logger.generate_summary_report()
        logger.write_log(f"✅ 简化版训练实验{hyper_exp}完成")

    summary_writer.close()


def run_mpi_main_training(mpi_rank, mpi_comm):
    """
    主控机器人训练函数 - MPI架构
    """
    print("🎯 启动主控机器人训练...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 主进程使用设备: {device}")

    sac(actor_critic=core.MLPActorCritic, device=device,
        mpi_rank=mpi_rank, mpi_size=mpi_comm.Get_size(), robot_index=0,
        is_main_robot=True, mpi_comm=mpi_comm)

    print("✅ 主控机器人训练完成!")


if __name__ == "__main__":
    from utils.mpi_utils.mpi_handler import (
        MPIHandler,
        ROLE_MAIN,
        ROLE_TARGET_PUBLISHER,
        ROLE_DYNAMIC_OBSTACLE
    )

    mpi_handler = MPIHandler()
    role, robot_index = mpi_handler.get_role_and_index()

    print(f"🚀 MPI进程启动: rank {mpi_handler.rank}/{mpi_handler.size}, 角色={role}, 机器人索引={robot_index}")

    if role == ROLE_MAIN:
        print("🎯 启动主控机器人训练进程")
        run_mpi_main_training(mpi_handler.rank, mpi_handler.mpi_comm)

    elif role == ROLE_TARGET_PUBLISHER:
        print("🎯 启动目标点发布进程")

    elif role == ROLE_DYNAMIC_OBSTACLE:
        print(f"🤖 启动动态障碍物进程 robot_{robot_index}...")
        run_mpi_dynamic_obstacle_control(robot_index, mpi_handler.rank, mpi_handler.mpi_comm)

    else:
        print(f"❌ 未知角色，进程 {mpi_handler.rank} 将退出")

    print(f"✅ MPI进程 rank {mpi_handler.rank} 结束")
