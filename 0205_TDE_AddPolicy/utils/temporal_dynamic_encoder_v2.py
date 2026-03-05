# -*- coding: utf-8 -*-
"""
Temporal Dynamic Encoder V2 - 改进版（去除reward预测）
===========================================================

改进点：
1. ✅ 添加损失权重平衡（pred_weight）
2. ✅ 添加梯度裁剪（gradient clipping）
3. ✅ 降低默认学习率
4. ✅ 添加Layer Normalization稳定训练
5. ✅ 支持Huber Loss（对outlier更鲁棒）
6. ✅ 添加AvgL1Norm（参考TD7）
7. ✅ 添加详细的训练统计

使用方法：
    from utils.temporal_dynamic_encoder_v2 import TDEManagerV2

    tde_manager = TDEManagerV2(
        device=device,
        lr=1e-4,                    # 降低学习率
        pred_loss_weight=1.0,       # embedding预测权重
        use_huber_loss=True,        # 使用Huber Loss
        grad_clip=1.0               # 梯度裁剪
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class TemporalDynamicEncoder(nn.Module):
    """时序动态编码器 - 与V1相同，不修改网络结构"""

    def __init__(self,
                 input_channels=10,
                 num_dynamic_points=90,
                 embedding_dim=64,
                 action_dim=2):
        super().__init__()

        self.input_channels = input_channels
        self.num_dynamic_points = num_dynamic_points
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        # CNN编码器
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, embedding_dim, kernel_size=1)
        self.pool = nn.MaxPool1d(num_dynamic_points)

        # State-Action Predictor
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, embedding_dim)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def encode(self, dynamic_points):
        """编码动态点云"""
        x = dynamic_points.transpose(1, 2)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        z = self.layer_norm(x)
        return z

    def predict_next(self, z_dynamic, action):
        """预测下一时刻embedding"""
        z_a = torch.cat([z_dynamic, action], dim=-1)
        z_pred = self.predictor(z_a)
        return z_pred

    def forward(self, dynamic_points, action=None):
        """完整前向传播"""
        z_dynamic = self.encode(dynamic_points)

        if action is not None:
            z_pred = self.predict_next(z_dynamic, action)
            return z_dynamic, z_pred

        return z_dynamic


class TDEManagerV2:
    """
    TDE管理器 V2 - 改进训练稳定性

    主要改进：
    1. 损失权重平衡
    2. 梯度裁剪
    3. Huber Loss选项
    4. AvgL1Norm
    5. 更详细的统计信息
    """

    def __init__(self,
                 device,
                 input_channels=10,
                 num_dynamic_points=90,
                 embedding_dim=64,
                 action_dim=2,
                 lr=1e-4,                    # 降低学习率
                 tau=0.005,
                 pred_loss_weight=1.0,       # embedding预测损失权重
                 use_huber_loss=True,        # 使用Huber Loss
                 huber_delta=1.0,            # Huber Loss的delta
                 grad_clip=1.0):             # 梯度裁剪阈值

        self.device = device
        self.tau = tau
        self.embedding_dim = embedding_dim
        self.num_dynamic_points = num_dynamic_points
        self.input_channels = input_channels

        # 损失权重
        self.pred_loss_weight = pred_loss_weight

        # 稳定性参数
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.grad_clip = grad_clip

        # 创建网络
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
        self.stats = {
            'pred_loss_history': [],
            'grad_norm_history': []
        }

        print(f"🔧 TDE V2 初始化:")
        print(f"   学习率: {lr}")
        print(f"   损失权重: pred={pred_loss_weight}")
        print(f"   Huber Loss: {use_huber_loss}")
        print(f"   梯度裁剪: {grad_clip}")

    def extract_dynamic_points(self, obs, pool_size=1800):
        """提取动态点"""
        batch_size = obs.shape[0]
        point_cloud = obs[:, :pool_size]
        point_cloud = point_cloud.view(batch_size, 180, self.input_channels)
        dynamic_points = point_cloud[:, 90:, :]
        return dynamic_points

    def _encode_dynamic(self, encoder, dynamic_points):
        z_dynamic = encoder.encode(dynamic_points)
        return avg_l1_norm(z_dynamic)

    def get_embeddings(self, obs, action=None, pool_size=1800):
        """获取embedding（推理时使用，梯度解耦）"""
        dynamic_points = self.extract_dynamic_points(obs, pool_size)

        with torch.no_grad():
            z_dynamic = self._encode_dynamic(self.tde_target, dynamic_points)

            if action is not None:
                z_pred = self.tde_target.predict_next(z_dynamic, action)
                return z_dynamic, z_pred

        return z_dynamic

    def predict_next_from_z(self, z_dynamic, action):
        """基于z_dynamic预测下一时刻embedding（梯度解耦）"""
        with torch.no_grad():
            return self.tde_target.predict_next(z_dynamic, action)

    def compute_loss(self, z_current, z_next_target, action):
        """
        计算损失 - 改进版

        改进：
        1. 损失权重平衡
        2. Huber Loss选项
        """
        # 预测下一时刻embedding
        z_pred = self.tde.predict_next(z_current, action)

        # ========== Embedding预测损失 ==========
        if self.use_huber_loss:
            pred_loss = F.smooth_l1_loss(z_pred, z_next_target.detach(),
                                          beta=self.huber_delta)
        else:
            pred_loss = F.mse_loss(z_pred, z_next_target.detach())

        # ========== 加权总损失 ==========
        total_loss = self.pred_loss_weight * pred_loss

        return total_loss, pred_loss

    def update(self, obs, obs_next, actions, pool_size=1800):
        """
        更新TDE - 改进版

        改进：
        1. 梯度裁剪
        2. 详细统计
        """
        # 提取动态点
        dynamic_points = self.extract_dynamic_points(obs, pool_size)
        dynamic_points_next = self.extract_dynamic_points(obs_next, pool_size)

        # 编码
        z_current = self._encode_dynamic(self.tde, dynamic_points)

        with torch.no_grad():
            z_next_target = self._encode_dynamic(self.tde_target, dynamic_points_next)

        # 计算损失
        total_loss, pred_loss = self.compute_loss(
            z_current, z_next_target, actions
        )

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.tde.parameters(),
                self.grad_clip
            )
        else:
            grad_norm = 0.0

        self.optimizer.step()

        # 软更新target
        self._soft_update()

        self.update_count += 1

        # 统计
        self.stats['pred_loss_history'].append(pred_loss.item())
        self.stats['grad_norm_history'].append(grad_norm if isinstance(grad_norm, float) else grad_norm.item())

        return {
            'tde_loss': total_loss.item(),
            'tde_pred_loss': pred_loss.item(),
            'tde_grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item()
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
        save_dict = {
            'tde': self.tde.state_dict(),
            'tde_target': self.tde_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'stats': self.stats
        }

        torch.save(save_dict, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.tde.load_state_dict(checkpoint['tde'])
        self.tde_target.load_state_dict(checkpoint['tde_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']

        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']

    def get_stats_summary(self):
        """获取训练统计摘要"""
        if len(self.stats['pred_loss_history']) == 0:
            return "No training data yet"

        recent = 100  # 最近100步

        summary = {
            'pred_loss_mean': np.mean(self.stats['pred_loss_history'][-recent:]),
            'grad_norm_mean': np.mean(self.stats['grad_norm_history'][-recent:])
        }

        return summary
