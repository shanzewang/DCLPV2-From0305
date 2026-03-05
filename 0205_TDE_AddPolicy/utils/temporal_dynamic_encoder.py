# -*- coding: utf-8 -*-
"""
Temporal Dynamic Encoder (TDE) - 时序动态编码器
===============================================

借鉴DARE论文的Temporal Dynamic Prediction (TDP)模块设计。

功能：
1. 编码动态障碍物特征 → z_dynamic
2. 预测下一时刻的动态embedding → z_pred
3. 预测下一时刻的reward → pred_reward

关键设计：
- 梯度解耦：TDE模块独立训练，不接收来自RL的梯度
- 使用辅助任务（状态预测+reward预测）训练
- 输出embedding用于增强Actor/Critic的输入

输入格式：
    动态点 [batch, 90, 10]: 后90个点的动态信息
    特征：[cos_α, sin_α, distance, vx, vy, is_dynamic, L1, L2, W, ttc_risk]

输出格式：
    z_dynamic: [batch, 64] - 当前时刻动态embedding
    z_pred: [batch, 64] - 预测的下一时刻embedding
    pred_reward: [batch, 1] - 预测的reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TemporalDynamicEncoder(nn.Module):
    """
    时序动态编码器 - 借鉴DARE的TDP模块

    核心功能：
    1. 编码动态障碍物特征
    2. 预测下一时刻的动态embedding
    3. 预测下一时刻的reward
    """

    def __init__(self,
                 input_channels=10,      # 每个点的特征维度
                 num_dynamic_points=90,  # 动态点数量
                 embedding_dim=64,       # embedding维度
                 action_dim=2):          # 动作维度 [v, w]
        super().__init__()

        self.input_channels = input_channels
        self.num_dynamic_points = num_dynamic_points
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        # ============= 动态特征编码器 (CNN) =============
        # PointNet风格: 10 -> 32 -> 64 -> 64
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=1)
        self.pool = nn.MaxPool1d(num_dynamic_points)

        # ============= State-Action Predictor g(z, a) =============
        # 预测下一时刻的动态embedding
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, embedding_dim)
        )

        # ============= Reward Decoder R(z) =============
        # 从预测的embedding解码reward
        self.reward_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        # ============= Layer Normalization (稳定训练) =============
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def encode(self, dynamic_points):
        """
        编码动态点云

        Args:
            dynamic_points: [batch, 90, 10] - 90个动态点，每点10维

        Returns:
            z_dynamic: [batch, 64] - 动态embedding
        """
        # [batch, 90, 10] -> [batch, 10, 90]
        x = dynamic_points.transpose(1, 2)

        # 三层卷积
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        # Global max pooling
        x = self.pool(x)  # [batch, 64, 1]
        z = x.squeeze(-1)  # [batch, 64]

        # Layer norm for stability
        z = self.layer_norm(z)

        return z

    def predict_next(self, z_dynamic, action):
        """
        预测下一时刻的dynamic embedding

        Args:
            z_dynamic: [batch, 64] - 当前动态embedding
            action: [batch, 2] - 执行的动作 [v, w]

        Returns:
            z_pred: [batch, 64] - 预测的下一时刻embedding
        """
        # Concatenate embedding and action
        z_a = torch.cat([z_dynamic, action], dim=-1)  # [batch, 66]

        # Predict next embedding
        z_pred = self.predictor(z_a)  # [batch, 64]

        return z_pred

    def predict_reward(self, z_pred):
        """
        从预测的embedding解码reward

        Args:
            z_pred: [batch, 64] - 预测的embedding

        Returns:
            pred_reward: [batch, 1] - 预测的reward
        """
        return self.reward_decoder(z_pred)

    def forward(self, dynamic_points, action=None):
        """
        完整前向传播

        Args:
            dynamic_points: [batch, 90, 10]
            action: [batch, 2] (optional)

        Returns:
            z_dynamic: [batch, 64]
            z_pred: [batch, 64] (if action provided)
            pred_reward: [batch, 1] (if action provided)
        """
        z_dynamic = self.encode(dynamic_points)

        if action is not None:
            z_pred = self.predict_next(z_dynamic, action)
            pred_reward = self.predict_reward(z_pred)
            return z_dynamic, z_pred, pred_reward

        return z_dynamic

    def compute_loss(self, z_current, z_next_target, action, reward_target):
        """
        计算TDE的训练损失

        Args:
            z_current: 当前embedding
            z_next_target: 下一时刻的真实embedding (会被detach)
            action: 执行的动作
            reward_target: 真实的reward

        Returns:
            total_loss: 总损失
            pred_loss: embedding预测损失
            reward_loss: reward预测损失
        """
        # 预测下一时刻embedding
        z_pred = self.predict_next(z_current, action)

        # Embedding预测损失 (对应DARE的Lpred)
        pred_loss = F.mse_loss(z_pred, z_next_target.detach())

        # Reward预测损失 (对应DARE的Lreward)
        pred_reward = self.predict_reward(z_pred)
        reward_loss = F.mse_loss(pred_reward.squeeze(-1), reward_target)

        total_loss = pred_loss + reward_loss

        return total_loss, pred_loss, reward_loss


class TDEManager:
    """
    TDE管理器 - 管理TDE的训练和推理

    功能：
    1. 维护TDE和target TDE (moving average)
    2. 提供训练接口
    3. 提供推理接口（带梯度解耦）
    """

    def __init__(self, device, input_channels=10, num_dynamic_points=90,
                 embedding_dim=64, action_dim=2, lr=3e-4, tau=0.005):
        """
        初始化TDE管理器

        Args:
            device: torch device
            input_channels: 每个点的特征维度
            num_dynamic_points: 动态点数量
            embedding_dim: embedding维度
            action_dim: 动作维度
            lr: 学习率
            tau: target网络软更新系数
        """
        self.device = device
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.num_dynamic_points = num_dynamic_points
        self.input_channels = input_channels

        # 创建TDE和target TDE
        self.tde = TemporalDynamicEncoder(
            input_channels=input_channels,
            num_dynamic_points=num_dynamic_points,
            embedding_dim=embedding_dim,
            action_dim=action_dim
        ).to(device)

        self.tde_target = copy.deepcopy(self.tde)

        # 优化器
        self.optimizer = torch.optim.Adam(self.tde.parameters(), lr=lr)

        # 统计
        self.update_count = 0

    def extract_dynamic_points(self, obs, pool_size=1800):
        """
        从观测中提取动态点

        Args:
            obs: [batch, 1808] - 完整观测
            pool_size: 点云数据大小 (180*10=1800)

        Returns:
            dynamic_points: [batch, 90, 10] - 后90个点（动态点）
        """
        batch_size = obs.shape[0]

        # 提取点云部分
        point_cloud = obs[:, :pool_size]  # [batch, 1800]

        # 重塑为 [batch, 180, 10]
        point_cloud = point_cloud.view(batch_size, 180, self.input_channels)

        # 取后90个点（动态点）
        dynamic_points = point_cloud[:, 90:, :]  # [batch, 90, 10]

        return dynamic_points

    def get_embeddings(self, obs, action=None, pool_size=1800):
        """
        获取动态embedding（用于Actor/Critic，梯度解耦）

        Args:
            obs: [batch, 1808] - 完整观测
            action: [batch, 2] - 动作 (optional)
            pool_size: 点云数据大小

        Returns:
            z_dynamic: [batch, 64] - 当前动态embedding (detached)
            z_pred: [batch, 64] - 预测embedding (detached, if action provided)
        """
        dynamic_points = self.extract_dynamic_points(obs, pool_size)

        with torch.no_grad():
            z_dynamic = self.tde_target.encode(dynamic_points)

            if action is not None:
                z_pred = self.tde_target.predict_next(z_dynamic, action)
                return z_dynamic, z_pred

        return z_dynamic

    def update(self, obs, obs_next, actions, rewards, pool_size=1800):
        """
        更新TDE

        Args:
            obs: [batch, 1808] - 当前观测
            obs_next: [batch, 1808] - 下一时刻观测
            actions: [batch, 2] - 动作
            rewards: [batch] - reward
            pool_size: 点云数据大小

        Returns:
            loss_info: dict with loss values
        """
        # 提取动态点
        dynamic_points = self.extract_dynamic_points(obs, pool_size)
        dynamic_points_next = self.extract_dynamic_points(obs_next, pool_size)

        # 编码当前和下一时刻
        z_current = self.tde.encode(dynamic_points)

        with torch.no_grad():
            z_next_target = self.tde_target.encode(dynamic_points_next)

        # 计算损失
        total_loss, pred_loss, reward_loss = self.tde.compute_loss(
            z_current, z_next_target, actions, rewards
        )

        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 软更新target网络
        self._soft_update()

        self.update_count += 1

        return {
            'tde_loss': total_loss.item(),
            'tde_pred_loss': pred_loss.item(),
            'tde_reward_loss': reward_loss.item()
        }

    def _soft_update(self):
        """软更新target网络"""
        for param, target_param in zip(self.tde.parameters(),
                                        self.tde_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        """保存模型"""
        torch.save({
            'tde': self.tde.state_dict(),
            'tde_target': self.tde_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.tde.load_state_dict(checkpoint['tde'])
        self.tde_target.load_state_dict(checkpoint['tde_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
