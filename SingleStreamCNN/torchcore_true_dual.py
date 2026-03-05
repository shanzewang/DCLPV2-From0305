# -*- coding: utf-8 -*-
"""
SAC神经网络架构 - 共享权重双流PointNet (Shared-Weight Dual-Stream)
================================================================

核心改进：将180个点分为两组，用**共享权重**的CNN分别处理：
    - 静态流: 前90个点（最近点，静态+动态混合）
    - 动态流: 后90个点（最近动态点）

与独立双流的区别：
    - 共享双流: 两个流用**同一套**卷积权重，但**分别池化**再融合
    - 独立双流: 两个流用**各自的**卷积权重

关键点 - 为什么共享权重+分别池化仍然优于原始方案：
    - 原始方案: 180个点 → 共享编码 → 对所有180点做MaxPool → 无法区分来源
    - 共享双流: 90个点×2 → 共享编码 → 分别对每组90点做MaxPool → 保留来源信息
    - 即使卷积权重相同，分别池化也能让网络知道"这个max来自静态点还是动态点"

优势：
    - 参数量少（仅比原始方案多一个融合层）
    - 共享权重提供正则化效果，防止过拟合
    - 仍然保留了分离池化的优势

输入格式：
    - 状态维度: 1808 = 180*10 + 8
    - 点云特征 [180, 10]: [cos_α, sin_α, distance, vx, vy, is_dynamic, L1, L2, W, ttc_risk]
        - 前90个点：最近点信息（按距离排序）
        - 后90个点：最近动态点信息（按距离排序）
    - 机器人状态 [8]: [d_g, theta_g, v, w, v_max, w_max, a_max, alpha_max]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

# ============= 全局常量 =============
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def init_weights_xavier(m):
    """Xavier均匀初始化"""
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def clip_but_pass_gradient2(x, l=EPS):
    """带梯度传递的下界裁剪"""
    clip_low = (x < l).to(x.dtype)
    return x + ((l - x) * clip_low).detach()


def new_relu(x, alpha_actv):
    """自定义激活函数: 1/(x + α + ε)"""
    r = torch.reciprocal(clip_but_pass_gradient2(x + alpha_actv, l=EPS))
    return r


def clip_but_pass_gradient(x, l=-1., u=1.):
    """双边梯度保持裁剪"""
    clip_up = (x > u).to(x.dtype)
    clip_low = (x < l).to(x.dtype)
    return x + (((u - x) * clip_up + (l - x) * clip_low)).detach()


class MLP(nn.Module):
    """MLP模块类"""
    def __init__(self, input_dim, hidden_sizes, activation=lambda x: F.leaky_relu(x, negative_slope=0.2), output_activation=None):
        super(MLP, self).__init__()
        self.activation = activation
        self.output_activation = output_activation

        layers = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            else:
                if self.output_activation is not None:
                    x = self.output_activation(x)
        return x


# =====================================================================
#  核心改进：共享权重双流 PointNet
# =====================================================================

class SharedEncoder(nn.Module):
    """
    共享的PointNet编码器 - 被两个流复用

    结构: input_channels -> 32 -> 64 -> out_channels
    注意：不包含MaxPool，池化在外部按流分别进行
    """
    def __init__(self, input_channels=10, out_channels=64):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [batch, input_channels, num_points]
        Returns:
            [batch, out_channels, num_points] - 逐点特征（未池化）
        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        return x


class SharedDualStreamCNNNet(nn.Module):
    """
    共享权重双流CNN网络

    架构：
        前90个点 → [共享编码器] → 逐点特征 → [MaxPool] → z_static  (64维)
        后90个点 → [共享编码器] → 逐点特征 → [MaxPool] → z_dynamic (64维)
                                                           ↓
                                              concat → [融合层] → output (128维)

    关键区别于混合方案：
        - 即使权重共享，前90和后90仍然**分别做MaxPool**
        - 混合方案: max(所有180个点) → 无法区分来源
        - 共享双流: max(前90) + max(后90) → 保留来源信息
    """
    def __init__(self, input_channels=10, num_points=180, activation=F.relu, output_activation=None):
        super(SharedDualStreamCNNNet, self).__init__()
        assert num_points == 180, "SharedDualStreamCNNNet requires 180 points (90 static + 90 dynamic)"

        # 共享的编码器（同一套权重处理两组点）
        self.shared_encoder = SharedEncoder(input_channels, out_channels=64)

        # 分别池化
        self.pool = nn.MaxPool1d(kernel_size=90)

        # 融合层: 64+64=128 -> 128
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x, y=None):
        """
        Args:
            x: [batch, 180, 10] - 已经过距离变换的点云
        Returns:
            [batch, 128]
        """
        # 显式分离两组点
        static_points = x[:, :90, :].transpose(1, 2)    # [batch, 10, 90]
        dynamic_points = x[:, 90:, :].transpose(1, 2)   # [batch, 10, 90]

        # 共享编码器处理两组点（同一套权重）
        static_feat = self.shared_encoder(static_points)    # [batch, 64, 90]
        dynamic_feat = self.shared_encoder(dynamic_points)   # [batch, 64, 90]

        # 分别池化（关键！不是对180个点一起池化）
        z_static = self.pool(static_feat).squeeze(-1)    # [batch, 64]
        z_dynamic = self.pool(dynamic_feat).squeeze(-1)   # [batch, 64]

        # 拼接 + 融合
        combined = torch.cat([z_static, z_dynamic], dim=-1)  # [batch, 128]
        output = self.fusion(combined)  # [batch, 128]

        return output


class SharedDualStreamCNNDense(nn.Module):
    """
    共享权重双流CNN特征提取器

    输出: [batch, 136] = CNN特征(128) + 机器人状态(8)
    """
    def __init__(self, input_channels=10, num_points=180,
                 activation=lambda x: F.leaky_relu(x, negative_slope=0.2),
                 output_activation=None):
        super(SharedDualStreamCNNDense, self).__init__()
        self.num_points = num_points
        self.input_channels = input_channels
        self.pool_size = num_points * input_channels  # 1800

        # 距离倒数变换参数
        self.alpha_actv2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # 共享权重双流CNN
        self.dual_cnn = SharedDualStreamCNNNet(input_channels=input_channels, num_points=num_points)

    def forward(self, x):
        """
        Args:
            x: [batch, 1808]
        Returns:
            [batch, 136]
        """
        x_input = x[:, 0:self.pool_size].view(-1, self.num_points, self.input_channels)

        x00 = new_relu(x_input[:, :, 2], self.alpha_actv2)
        x_input = torch.cat([
            x_input[:, :, 0:2],
            x00.unsqueeze(-1),
            x_input[:, :, 3:10]
        ], dim=-1)

        cnn_feat = self.dual_cnn(x_input)  # [batch, 128]

        return torch.cat([cnn_feat, x[:, self.pool_size:]], dim=-1)  # [batch, 136]


def count_vars(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_log_gaussian(mu_t, log_sig_t, t):
    """高斯混合模型概率计算"""
    normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)
    quadratic = -0.5 * torch.sum(normalized_dist_t ** 2, dim=-1)
    log_z = torch.sum(log_sig_t, dim=-1)
    D_t = float(mu_t.shape[-1])
    log_z += 0.5 * D_t * np.log(2 * np.pi)
    log_p = quadratic - log_z
    return log_p


class MLPGaussianPolicy(nn.Module):
    """
    高斯混合策略网络 - 共享权重双流版本
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=[128,128,128,128],
                 input_channels=10, num_points=180,
                 activation=lambda x: F.leaky_relu(x, negative_slope=0.2),
                 output_activation=None):
        super(MLPGaussianPolicy, self).__init__()

        self.k = 4
        self.act_dim = action_dim
        self.activation = activation
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.num_points = num_points
        self.pool_size = num_points * input_channels

        self.alpha_actv1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # 共享权重双流CNN
        self.cnn = SharedDualStreamCNNNet(input_channels=input_channels, num_points=num_points)

        # MLP: 128 + 8 = 136
        self.mlp = MLP(136, hidden_sizes, activation, activation)
        self.gmm_layer = nn.Linear(hidden_sizes[-1], (self.act_dim*2+1)*self.k)

    def forward(self, x):
        batch_size = x.shape[0]

        x_input = x[:, 0:self.pool_size].view(-1, self.num_points, self.input_channels)

        x0 = new_relu(x_input[:, :, 2], self.alpha_actv1)
        x_input = torch.cat([
            x_input[:, :, 0:2],
            x0.unsqueeze(-1),
            x_input[:, :, 3:10]
        ], dim=-1)

        w_input = x[:, self.pool_size:self.pool_size+8].view(-1, 8)

        cnn_net = self.cnn(x_input, w_input)  # [batch, 128]

        y = torch.cat([cnn_net, x[:, self.pool_size:]], dim=-1)  # [batch, 136]

        net = self.mlp(y)

        w_and_mu_and_logsig_t = self.gmm_layer(net)
        w_and_mu_and_logsig_t = w_and_mu_and_logsig_t.view(-1, self.k, 2*self.act_dim+1)

        log_w_t = w_and_mu_and_logsig_t[..., 0]
        mu_t = w_and_mu_and_logsig_t[..., 1:1+self.act_dim]
        log_sig_t = w_and_mu_and_logsig_t[..., 1+self.act_dim:]

        log_sig_t = torch.tanh(log_sig_t)
        log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
        xz_sigs_t = torch.exp(log_sig_t)

        z_t = torch.multinomial(torch.softmax(log_w_t, dim=-1), num_samples=1)

        batch_indices = torch.arange(batch_size, device=x.device)
        xz_mu_t = mu_t[batch_indices, z_t.squeeze(-1)]
        xz_sig_t = xz_sigs_t[batch_indices, z_t.squeeze(-1)]

        epsilon = torch.randn((batch_size, self.act_dim), device=x.device)
        x_t = xz_mu_t + xz_sig_t * epsilon

        log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t.unsqueeze(1))
        log_p_x_t_numerator = torch.logsumexp(log_p_xz_t + log_w_t, dim=1)
        log_p_x_t_denominator = torch.logsumexp(log_w_t, dim=1)
        log_p_x_t = log_p_x_t_numerator - log_p_x_t_denominator

        logp_pi = log_p_x_t
        return xz_mu_t, x_t, logp_pi


def apply_squashing_func(mu, pi, logp_pi):
    """Tanh压缩函数及雅可比校正"""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    logp_pi -= torch.sum(torch.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), dim=1)
    return mu, pi, logp_pi


class MLPActorCritic(nn.Module):
    """
    Actor-Critic网络 - 共享权重双流版本

    与独立双流版本的区别：
        - 两个流共享卷积权重（参数量更少）
        - 分别池化（保留来源信息）
        - 输出维度不变，与训练文件完全兼容
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128,128,128),
                 input_channels=10, num_points=180,
                 activation=lambda x: F.leaky_relu(x, negative_slope=0.2),
                 output_activation=None):
        super(MLPActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.input_channels = input_channels
        self.num_points = num_points

        # Actor
        self.policy = MLPGaussianPolicy(state_dim, action_dim, list(hidden_sizes),
                                        input_channels=input_channels,
                                        num_points=num_points,
                                        activation=activation,
                                        output_activation=output_activation)

        # Critic: 共享权重双流CNN
        self.cnn_dense = SharedDualStreamCNNDense(input_channels=input_channels,
                                                   num_points=num_points,
                                                   activation=activation,
                                                   output_activation=None)

        # Q网络: 128 + 8 + 2 = 138
        q_input_dim = 128 + 8 + action_dim
        self.q1 = MLP(q_input_dim, list(hidden_sizes) + [1], activation, None)
        self.q2 = MLP(q_input_dim, list(hidden_sizes) + [1], activation, None)

    def forward(self, x, a=None):
        mu, pi, logp_pi = self.policy(x)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        y = self.cnn_dense(x)  # [batch, 136]

        q1_pi = self.q1(torch.cat([y, pi], dim=-1)).squeeze(-1)
        q2_pi = self.q2(torch.cat([y, pi], dim=-1)).squeeze(-1)

        if a is not None:
            q1 = self.q1(torch.cat([y, a], dim=-1)).squeeze(-1)
            q2 = self.q2(torch.cat([y, a], dim=-1)).squeeze(-1)
            return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
        else:
            return mu, pi, logp_pi, q1_pi, q2_pi
