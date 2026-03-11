import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
import argparse
from torch import amp
import json
import pickle
import sys

# 设置随机种子
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 工具函数
def convert_numpy_to_python(obj):
    """递归地将numpy类型转换为Python原生类型"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


#######################################################
# 数据集和工具函数
#######################################################

class SSTDataset(Dataset):
    """SST数据集 - 用于7天输入7天输出"""
    def __init__(self, X, y, ocean_mask):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.ocean_mask = torch.FloatTensor(ocean_mask.astype(np.float32))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_7day_sequences(data_dict, split='train'):
    """创建7天输入7天输出的训练序列"""
    sst_data = data_dict['sst_normalized']
    time_split = data_dict['time_split']
    ocean_mask = data_dict['ocean_mask']
    
    # 映射分割值
    split_mapping = {'train': 0, 'val': 1, 'test': 2}
    split_value = split_mapping[split]
    
    # 设置滑动窗口参数
    input_days = 7
    output_days = 7
    
    X, y = [], []
    
    # 使用滑动窗口创建序列
    for i in range(len(sst_data) - input_days - output_days + 1):
        last_output_time = i + input_days + output_days - 1
        
        # 检查序列的最后一个时间步是否在指定的分割中
        if time_split[last_output_time] == split_value:
            input_seq = sst_data[i:i + input_days]
            output_seq = sst_data[i + input_days:i + input_days + output_days]
            
            X.append(input_seq)
            y.append(output_seq)
    
    print(f"创建{split}集序列: {len(X)}")
    return np.array(X), np.array(y)


def load_preprocessed_data(file_path):
    """加载预处理后的数据"""
    print(f"加载数据: {file_path}")

    ds = xr.open_dataset(file_path)
    result = {
        'sst_normalized': ds['sst_normalized'].values,
        'ocean_mask': ds['ocean_mask'].values.astype(bool),
        'time_split': ds['time_split'].values,
        'sst_original': ds['sst_original'].values,
        'coords': {
            'time': ds.coords['time'].values,
            'lat': ds.coords['latitude'].values,
            'lon': ds.coords['longitude'].values,
        },
        'metadata': dict(ds.attrs)
    }
    ds.close()

    print(f"✓ 数据加载完成: {result['sst_normalized'].shape}")
    return result


def load_and_prepare_data(data_file):
    """加载并准备7天输入7天输出的数据"""
    print("加载预处理数据...")
    
    data_dict = load_preprocessed_data(data_file)
    
    # 使用自定义函数创建7天输入，7天输出的序列
    train_X, train_y = create_7day_sequences(data_dict, 'train')
    val_X, val_y = create_7day_sequences(data_dict, 'val')
    test_X, test_y = create_7day_sequences(data_dict, 'test')
    
    # 添加通道维度
    train_X = np.expand_dims(train_X, axis=2)  # (N, 7, 1, H, W)
    train_y = np.expand_dims(train_y, axis=2)  # (N, 7, 1, H, W)
    val_X = np.expand_dims(val_X, axis=2)
    val_y = np.expand_dims(val_y, axis=2)
    test_X = np.expand_dims(test_X, axis=2)
    test_y = np.expand_dims(test_y, axis=2)
    
    ocean_mask = data_dict['ocean_mask']
    
    print(f"训练集: X{train_X.shape}, y{train_y.shape}")
    print(f"验证集: X{val_X.shape}, y{val_y.shape}")
    print(f"测试集: X{test_X.shape}, y{test_y.shape}")
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y), ocean_mask, data_dict


def ocean_masked_loss(pred, target, ocean_mask):
    """海洋掩膜损失函数"""
    # 扩展mask维度以匹配pred和target
    batch_size, days, channels, height, width = pred.size()
    mask = ocean_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
    mask = mask.expand(batch_size, days, channels, height, width)  # (batch, days, channels, H, W)
    
    # 应用掩膜
    masked_pred = pred * mask
    masked_target = target * mask
    
    # 计算MSE损失
    mse = nn.functional.mse_loss(masked_pred, masked_target, reduction='none')
    
    # 计算有效像素并求平均
    valid_pixels = mask.sum()
    if valid_pixels > 0:
        return mse.sum() / valid_pixels
    else:
        return mse.mean()


def calculate_metrics(preds, targets, ocean_mask, data_min, data_max):
    """计算评估指标 - 使用MSE, MAE, R2"""
    output_days = preds.shape[1]
    metrics_by_day = []
    
    # 确保掩膜是浮点类型
    ocean_mask = ocean_mask.astype(np.float32)
    
    for day in range(output_days):
        # 获取当天预测和目标
        day_preds = preds[:, day, 0]  # (B, H, W)
        day_targets = targets[:, day, 0]  # (B, H, W)
        
        # 创建批次维度的掩膜
        batch_size = day_preds.shape[0]
        batch_mask = np.repeat(np.expand_dims(ocean_mask, axis=0), batch_size, axis=0)  # (B, H, W)
        
        # 应用掩膜（元素乘法而不是布尔索引）
        masked_preds = day_preds * batch_mask
        masked_targets = day_targets * batch_mask
        
        # 计算有效像素数
        valid_pixels = np.sum(batch_mask)
        
        # 计算指标
        squared_diff = (masked_preds - masked_targets) ** 2
        abs_diff = np.abs(masked_preds - masked_targets)
        
        # 使用有效像素计算均值
        mse = np.sum(squared_diff) / valid_pixels
        rmse = np.sqrt(mse)
        mae = np.sum(abs_diff) / valid_pixels
        
        # 计算R2
        masked_mean_target = np.sum(masked_targets) / valid_pixels
        ss_total = np.sum((masked_targets - masked_mean_target * batch_mask) ** 2)
        ss_residual = np.sum(squared_diff)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # 转换回摄氏度
        rmse_celsius = rmse * (data_max - data_min)
        mae_celsius = mae * (data_max - data_min)
        
        day_metrics = {
            'rmse': rmse,
            'rmse_celsius': rmse_celsius,
            'mae': mae,
            'mae_celsius': mae_celsius,
            'r2': r2
        }
        metrics_by_day.append(day_metrics)
    
    # 计算平均指标
    avg_metrics = {
        'rmse': np.mean([m['rmse'] for m in metrics_by_day]),
        'rmse_celsius': np.mean([m['rmse_celsius'] for m in metrics_by_day]),
        'mae': np.mean([m['mae'] for m in metrics_by_day]),
        'mae_celsius': np.mean([m['mae_celsius'] for m in metrics_by_day]),
        'r2': np.mean([m['r2'] for m in metrics_by_day])
    }
    
    return {'by_day': metrics_by_day, 'average': avg_metrics}


def visualize_predictions(preds, targets, ocean_mask, data_min, data_max, num_samples=2, save_dir="./results/ekf_predrnn"):
    """可视化预测结果"""
    output_days = preds.shape[1]
    
    for i in range(num_samples):
        fig, axes = plt.subplots(output_days, 3, figsize=(15, 4*output_days))
        
        for day in range(output_days):
            pred = preds[i, day, 0]  # (H, W)
            target = targets[i, day, 0]  # (H, W)
            diff = np.abs(pred - target)
            
            # 转换回摄氏度
            pred_celsius = pred * (data_max - data_min) + data_min
            target_celsius = target * (data_max - data_min) + data_min
            diff_celsius = diff * (data_max - data_min)
            
            # 预测结果
            im1 = axes[day, 0].imshow(pred_celsius, cmap='RdYlBu_r')
            axes[day, 0].set_title(f'Day {day+1} - EKF-PredrNN 预测')
            axes[day, 0].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im1, ax=axes[day, 0], shrink=0.8, label='°C')
            
            # 真实值
            im2 = axes[day, 1].imshow(target_celsius, cmap='RdYlBu_r')
            axes[day, 1].set_title(f'Day {day+1} - 真实')
            axes[day, 1].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im2, ax=axes[day, 1], shrink=0.8, label='°C')
            
            # 差异
            im3 = axes[day, 2].imshow(diff_celsius, cmap='Reds')
            axes[day, 2].set_title(f'Day {day+1} - 误差')
            axes[day, 2].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im3, ax=axes[day, 2], shrink=0.8, label='°C')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'ekf_predrnn_7day_predictions_sample_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存预测可视化图到 {save_path}")
        plt.close()


def plot_prediction_trend(metrics, save_dir="./results/ekf_predrnn"):
    """绘制7天预测趋势"""
    days = list(range(1, 8))
    rmse_values = [metrics['by_day'][i]['rmse_celsius'] for i in range(7)]
    mae_values = [metrics['by_day'][i]['mae_celsius'] for i in range(7)]
    r2_values = [metrics['by_day'][i]['r2'] for i in range(7)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制RMSE和MAE趋势
    ax1.plot(days, rmse_values, 'o-', color='blue', label='RMSE')
    ax1.plot(days, mae_values, 's-', color='red', label='MAE')
    ax1.set_xlabel('预测天数')
    ax1.set_ylabel('误差 (°C)')
    ax1.set_title('EKF-PredrNN RMSE和MAE随预测天数的变化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制R2趋势
    ax2.plot(days, r2_values, 'd-', color='green', label='R²')
    ax2.set_xlabel('预测天数')
    ax2.set_ylabel('R²')
    ax2.set_title('EKF-PredrNN R²随预测天数的变化')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "prediction_trend_ekf_predrnn_7day.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存预测趋势图到 {save_path}")
    plt.close()


#######################################################
# 扩展卡尔曼滤波层
#######################################################

class DiffKalmanFilterLayer(nn.Module):
    """可微分扩展卡尔曼滤波层"""

    def __init__(self, state_dim):
        super(DiffKalmanFilterLayer, self).__init__()
        self.state_dim = state_dim

        # 1. 状态转移参数(使用低秩分解减少参数量)
        rank = min(state_dim, 24)  # 低秩表示
        self.F_low = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
        self.F_high = nn.Parameter(torch.randn(rank, state_dim) * 0.01)

        # 2. 非线性状态转移网络
        self.nonlinear_net = nn.Sequential(
            nn.Linear(state_dim, 128),     # 从64增加到128
            nn.LayerNorm(128),             # 添加层归一化
            nn.LeakyReLU(),                # 替换Tanh为LeakyReLU
            nn.Dropout(0.1),               # 添加Dropout
            nn.Linear(128, state_dim),
            nn.Tanh()
        )
        # 状态依赖的噪声估计
        self.state_to_noise = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Sigmoid()                   # 将噪声与状态关联
        )
        self.nonlinear_weight = nn.Parameter(torch.tensor(0.1))

        # 3. 噪声协方差估计(对角)
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim))  # 过程噪声
        self.log_R_diag = nn.Parameter(torch.zeros(state_dim))  # 观测噪声

    def forward(self, state, P, observation):
        """
        forward - 一步卡尔曼滤波更新
        state: [B, state_dim] - 当前状态
        P: [B, state_dim, state_dim] - 当前状态协方差
        observation: [B, state_dim] - 观测
        """
        batch_size = state.shape[0]
        device = state.device

        # 构建状态转移矩阵
        F = torch.matmul(self.F_low, self.F_high)
        # 加入稳定性约束 - 单位矩阵偏置
        F = F + torch.eye(self.state_dim, device=device)
        F = F.unsqueeze(0).expand(batch_size, -1, -1)

        # 构建噪声协方差
        Q_diag = torch.exp(self.log_Q_diag)
        R_diag = torch.exp(self.log_R_diag)
        Q = torch.diag_embed(Q_diag).unsqueeze(0).expand(batch_size, -1, -1)
        R = torch.diag_embed(R_diag).unsqueeze(0).expand(batch_size, -1, -1)

        # 1. 预测步骤
        # 线性状态转移
        state_pred = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1)

        # 添加非线性动态
        nonlinear_contrib = self.nonlinear_net(state)
        state_pred = state_pred + torch.sigmoid(self.nonlinear_weight) * nonlinear_contrib

        # 预测协方差
        P_pred = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q

        # 2. 更新步骤
        # 简化观测矩阵(单位矩阵)
        H = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # 计算新息(innovation)
        innovation = observation - state_pred

        # 新息协方差
        S = torch.bmm(torch.bmm(H, P_pred), H.transpose(1, 2)) + R

        # 增强数值稳定性
        S_reg = S + 1e-5 * torch.eye(self.state_dim, device=device).unsqueeze(0)
        S_inv = torch.inverse(S_reg)
        K = torch.bmm(torch.bmm(P_pred, H.transpose(1, 2)), S_inv)

        # 状态更新
        state_updated = state_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)

        # 协方差更新(Joseph表达形式,数值更稳定)
        I = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        term1 = I - torch.bmm(K, H)
        P_updated = torch.bmm(torch.bmm(term1, P_pred), term1.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K, R), K.transpose(1, 2))

        return state_updated, P_updated, K


#######################################################
# 融合EKF的STLSTM单元
#######################################################

class DKFSTLSTMCell(nn.Module):
    """融合可微分卡尔曼滤波的时空LSTM单元"""

    def __init__(self, hidden_channels, state_dim, kernel_size=3, img_size=(32, 32)):
        super(DKFSTLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.state_dim = state_dim
        self.img_size = img_size
        padding = kernel_size // 2

        # 基础STLSTM组件
        self.conv_gates = nn.Conv2d(hidden_channels * 3, hidden_channels * 4,
                                    kernel_size=kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=kernel_size, padding=padding)

        # 状态编码器 - 将特征图编码为低维状态
        self.state_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),    # 适应32×32→16×16
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, state_dim),  # 8192 → state_dim
            nn.LayerNorm(state_dim)               # 添加归一化
        )

        # 状态解码器 - 将低维状态解码回特征图
        self.state_decoder = nn.Sequential(
            nn.Linear(state_dim, 32 * 16 * 16),
            nn.LeakyReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.Upsample(size=img_size, mode='bilinear', align_corners=False),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
            nn.LeakyReLU()
        )

        # 可微分卡尔曼滤波层
        self.kf_layer = DiffKalmanFilterLayer(state_dim)

        # 混合权重(可学习)
        self.kf_alpha = nn.Parameter(torch.tensor(0.3))

        # 残差连接权重
        self.res_weight = nn.Parameter(torch.tensor(0.1))

        # 预先创建输入通道调整层 - 假设可能的输入通道为1、3和hidden_channels
        self.input_conv_1 = nn.Conv2d(1, hidden_channels, kernel_size=1)
        self.input_conv_3 = nn.Conv2d(3, hidden_channels, kernel_size=1)

    def forward(self, x, h, c, m, kf_state=None, kf_cov=None):
        # 确保输入通道匹配
        if x.size(1) != self.hidden_channels:
            if x.size(1) == 1:
                x = self.input_conv_1(x)
            elif x.size(1) == 3:
                x = self.input_conv_3(x)

        # 获取x的空间尺寸作为标准
        _, _, H_x, W_x = x.shape
        _, _, H_h, W_h = h.shape

        # 如果空间尺寸不匹配，调整h、c和m
        if H_x != H_h or W_x != W_h:
            h = F.interpolate(h, size=(H_x, W_x), mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=(H_x, W_x), mode='bilinear', align_corners=False)
            m = F.interpolate(m, size=(H_x, W_x), mode='bilinear', align_corners=False)

        # 保存原始h用于残差连接
        h_orig = h

        # 1. 基础STLSTM更新
        combined = torch.cat([x, h, m], dim=1)
        gates = self.conv_gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)  # 候选记忆单元
        o = torch.sigmoid(o)  # 输出门

        c_next = f * c + i * g  # 记忆单元更新
        m_next = torch.sigmoid(self.conv_m(c_next)) * torch.tanh(c_next)  # 更新时空记忆
        h_lstm = o * torch.tanh(c_next)  # 基础LSTM隐状态

        # 2. 卡尔曼滤波状态更新
        batch_size = h.shape[0]
        if kf_state is None:
            kf_state = torch.zeros(batch_size, self.state_dim, device=h.device)
        if kf_cov is None:
            kf_cov = torch.eye(self.state_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(h.device) * 0.1

        # 编码隐状态到状态空间
        encoded_h = self.state_encoder(h_lstm)

        # 使用卡尔曼滤波更新状态
        kf_state_next, kf_cov_next, _ = self.kf_layer(kf_state, kf_cov, encoded_h)

        # 解码回特征空间
        decoded_state = self.state_decoder(kf_state_next)

        # 确保解码后的状态大小匹配
        if decoded_state.shape[2:] != h_lstm.shape[2:]:
            decoded_state = F.interpolate(
                decoded_state,
                size=(h_lstm.shape[2], h_lstm.shape[3]),
                mode='bilinear',
                align_corners=False
            )

        # 3. 混合LSTM和卡尔曼滤波结果
        alpha = torch.sigmoid(self.kf_alpha)
        h_next = h_lstm * (1 - alpha) + decoded_state * alpha
        
        # 4. 添加残差连接
        res_scale = torch.sigmoid(self.res_weight)
        h_next = h_next + res_scale * h_orig

        return h_next, c_next, m_next, kf_state_next, kf_cov_next


#######################################################
# 改进的EKF-PredRNN主模型 - 只使用一次降采样
#######################################################

class EKFPredRNN(nn.Module):
    """改进的EKF-PredRNN模型 - 只使用一次降采样"""

    def __init__(self, hidden_channels=64, state_dim=32, num_layers=3, output_days=7):
        super(EKFPredRNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.output_days = output_days
        self.img_size = (32, 32)  # 降采样后的尺寸

        # 编码器 - 仅一次降采样 64×64 → 32×32
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),      # 64×64 → 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, hidden_channels, 3, padding=1),   # 32×32 → 32×32
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 融合EKF的ST-LSTM单元
        self.st_lstm_cells = nn.ModuleList([
            DKFSTLSTMCell(hidden_channels, state_dim, kernel_size=3, img_size=self.img_size)
            for _ in range(num_layers)
        ])

        # 流记忆模块
        self.memory_flow_cells = nn.ModuleList([
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1)
            for _ in range(num_layers - 1)
        ])

        # 时间嵌入 - 适应32×32分辨率
        self.time_embedding = nn.Parameter(torch.zeros(1, 14, hidden_channels, 32, 32))  # 7+7天，32×32
        nn.init.xavier_uniform_(self.time_embedding)

        # 单步解码器 - 从32×32恢复到64×64
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),          # 32×32 → 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32×32 → 64×64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),                        # 64×64 → 64×64
            nn.Sigmoid()
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_sequence(self, x):
        """编码输入序列并返回最终状态"""
        B, T, C, H, W = x.shape
        device = x.device

        # 初始化隐状态和记忆 - 32×32
        H_encoded, W_encoded = 32, 32  # 编码后的尺寸

        h_list = [torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device)
                for _ in range(self.num_layers)]
        c_list = [torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device)
                for _ in range(self.num_layers)]
        m_list = [torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device)
                for _ in range(self.num_layers)]
        
        # 初始化卡尔曼状态
        kf_states = [None] * self.num_layers
        kf_covs = [None] * self.num_layers

        # 流记忆 - 32×32尺寸
        memory_flows = [torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device)
                        for _ in range(self.num_layers - 1)]

        # 处理序列
        for t in range(T):
            # 提取当前帧并编码
            x_t = x[:, t, :, :, :]  # [B, C, H, W]
            x_t = self.encoder(x_t)

            # 通过所有层
            for l in range(self.num_layers):
                h, c, m = h_list[l], c_list[l], m_list[l]
                kf_state, kf_cov = kf_states[l], kf_covs[l]

                # 添加时间嵌入
                if t < self.time_embedding.size(1):
                    time_emb = self.time_embedding[:, t]  # [1, hidden_channels, 32, 32]
                    x_t = x_t + time_emb

                # 应用流记忆
                if l > 0 and t > 0:
                    flow_input = torch.cat([x_t, memory_flows[l - 1]], dim=1)
                    flow = self.memory_flow_cells[l - 1](flow_input)
                    memory_flows[l - 1] = flow
                    x_t = x_t + 0.2 * flow

                # EKF-STLSTM更新
                h_next, c_next, m_next, kf_state_next, kf_cov_next = self.st_lstm_cells[l](
                    x_t, h, c, m, kf_state, kf_cov
                )

                # 更新所有状态
                h_list[l], c_list[l], m_list[l] = h_next, c_next, m_next
                kf_states[l], kf_covs[l] = kf_state_next, kf_cov_next

                # 下一层的输入
                x_t = h_next

        return h_list, c_list, m_list, kf_states, kf_covs, memory_flows

    def _decode_step(self, h_list, c_list, m_list, kf_states, kf_covs, memory_flows, decoder_input, t):
        """单步解码 - 生成一天的预测"""
        # 编码输入
        x_t = self.encoder(decoder_input)
        
        # 通过所有层
        for l in range(self.num_layers):
            h, c, m = h_list[l], c_list[l], m_list[l]
            kf_state, kf_cov = kf_states[l], kf_covs[l]
            
            # 添加时间嵌入 - 为解码步骤添加时间索引偏移
            if t + 7 < self.time_embedding.size(1):  # 7是输入序列长度
                time_emb = self.time_embedding[:, t + 7]  # [1, hidden_channels, 32, 32]
                x_t = x_t + time_emb
            
            # 应用流记忆
            if l > 0:
                flow_input = torch.cat([x_t, memory_flows[l - 1]], dim=1)
                flow = self.memory_flow_cells[l - 1](flow_input)
                memory_flows[l - 1] = flow
                x_t = x_t + 0.2 * flow
            
            # EKF-STLSTM更新
            h_next, c_next, m_next, kf_state_next, kf_cov_next = self.st_lstm_cells[l](
                x_t, h, c, m, kf_state, kf_cov
            )
            
            # 更新状态
            h_list[l], c_list[l], m_list[l] = h_next, c_next, m_next
            kf_states[l], kf_covs[l] = kf_state_next, kf_cov_next
            
            # 下一层的输入
            x_t = h_next
        
        # 从最终隐藏状态解码输出
        output = self.decoder(h_list[-1])
        
        return output, h_list, c_list, m_list, kf_states, kf_covs, memory_flows

    def forward(self, x, target_tensor=None, teacher_forcing_ratio=0.0):
        """
        前向传播 - 支持教师强制的迭代解码版本
        参数:
            x: 输入序列 [B, T, C, H, W]
            target_tensor: 目标序列 [B, T, C, H, W]
            teacher_forcing_ratio: 教师强制率，范围[0,1]
        """
        # 编码输入序列
        h_list, c_list, m_list, kf_states, kf_covs, memory_flows = self._encode_sequence(x)
        
        # 准备解码
        outputs = []
        decoder_input = x[:, -1, :, :, :]  # 最后一个输入帧
        
        # 按照论文方式实现教师强制 - 对整个序列使用相同的决策
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        # 逐步解码生成预测序列
        for t in range(self.output_days):
            # 单步解码
            output, h_list, c_list, m_list, kf_states, kf_covs, memory_flows = self._decode_step(
                h_list, c_list, m_list, kf_states, kf_covs, memory_flows, decoder_input, t
            )
            
            # 保存当前预测
            outputs.append(output)
            
            # 更新下一步的输入
            if use_teacher_forcing and target_tensor is not None and t < self.output_days - 1:
                decoder_input = target_tensor[:, t, :, :, :]
            else:
                decoder_input = output
        
        # 拼接所有输出
        outputs = torch.stack(outputs, dim=1)  # [B, output_days, 1, H, W]
        
        return outputs


#######################################################
# 训练和评估函数
#######################################################

def evaluate_model(model, test_loader, ocean_mask, data_min, data_max, device='cuda'):
    """评估模型"""
    model.eval()
    model = model.to(device)
    ocean_mask = ocean_mask.to(device)
    
    test_loss = 0.0
    test_batches = 0
    all_preds = []
    all_targets = []
    
    t0 = time.time()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing EKF-PredrNN'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播 - 不使用教师强制
            pred = model(batch_x, teacher_forcing_ratio=0.0)
            
            # 计算损失
            loss = ocean_masked_loss(pred, batch_y, ocean_mask)
            
            test_loss += loss.item()
            test_batches += 1
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_test_loss = test_loss / test_batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    metrics = calculate_metrics(all_preds, all_targets, ocean_mask.cpu().numpy(), data_min, data_max)
    
    print(f'***** EKF-PredrNN Test Results *****')
    print(f'MSE: {avg_test_loss:.6f}')
    
    # 打印每天的结果
    for day in range(all_preds.shape[1]):
        day_metrics = metrics['by_day'][day]
        print(f"Day {day+1} 预测性能:")
        print(f"  RMSE: {day_metrics['rmse_celsius']:.4f}°C")
        print(f"  MAE: {day_metrics['mae_celsius']:.4f}°C")
        print(f"  R2: {day_metrics['r2']:.4f}")
    
    print(f'平均: RMSE: {metrics["average"]["rmse_celsius"]:.4f}°C, '
          f'MAE: {metrics["average"]["mae_celsius"]:.4f}°C, '
          f'R2: {metrics["average"]["r2"]:.6f}')
    print(f'Test Time: {time.time() - t0:.1f}s')
    
    return avg_test_loss, all_preds, all_targets, metrics


def train_model(model, train_loader, val_loader, ocean_mask, data_min, data_max, 
                num_epochs=50, learning_rate=0.001, device='cuda', save_dir="./results/ekf_predrnn"):
    model = model.to(device)
    ocean_mask = ocean_mask.to(device)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.1, patience=10, verbose=True)
    
    # 混合精度训练
    scaler = amp.GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_patience = 20
    no_improve_epochs = 0
    
    print("开始训练 EKF-PredrNN...")
    
    for epoch in range(num_epochs):
        t0 = time.time()
        
        # 按每个epoch更新教师强制率 - 采用线性衰减策略
        teacher_forcing_ratio = max(0.0, 1.0 - epoch * 0.02)  # 100个epoch内完成衰减
        
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 前向传播 - 使用当前教师强制率
                pred = model(batch_x, target_tensor=batch_y, teacher_forcing_ratio=teacher_forcing_ratio)
                
                # 计算损失
                loss = ocean_masked_loss(pred, batch_y, ocean_mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # 前向传播 - 验证时不使用教师强制
                pred = model(batch_x, teacher_forcing_ratio=0.0)
                
                loss = ocean_masked_loss(pred, batch_y, ocean_mask)
                
                val_loss += loss.item()
                val_batches += 1
                
                val_predictions.append(pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 计算评估指标
        val_preds = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        metrics = calculate_metrics(val_preds, val_targets, ocean_mask.cpu().numpy(), data_min, data_max)
        
        # 打印本轮结果
        print(f'Epoch {epoch+1}/{num_epochs} (Time: {time.time()-t0:.1f}s):')
        print(f'  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        print(f'  Teacher Forcing Ratio: {teacher_forcing_ratio:.3f}')
        
        # 打印每天详细指标
        print('  每天预测误差:')
        for day in range(val_preds.shape[1]):
            day_metrics = metrics['by_day'][day]
            print(f'    Day {day+1}: RMSE = {day_metrics["rmse_celsius"]:.4f}°C, '
                 f'MAE = {day_metrics["mae_celsius"]:.4f}°C, '
                 f'R2 = {day_metrics["r2"]:.4f}')
        
        print(f'  平均: RMSE = {metrics["average"]["rmse_celsius"]:.4f}°C, '
              f'MAE = {metrics["average"]["mae_celsius"]:.4f}°C, '
              f'R2 = {metrics["average"]["r2"]:.4f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_ekf_predrnn_7day_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'  [保存最佳模型] 验证损失: {best_val_loss:.6f} 到 {model_path}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"早停：连续{early_stop_patience}轮无改善")
                break
    
    return train_losses, val_losses


def main(data_file='sst_final.nc', save_dir="./results/ekf_predrnn"):
    """主函数 - EKF-PredrNN实现"""
    print("=" * 60)
    print("EKF-PredrNN 海温预测模型 - 7天输入7天预测")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到: {save_dir}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    (train_X, train_y), (val_X, val_y), (test_X, test_y), ocean_mask, data_dict = load_and_prepare_data(data_file)
    
    # 加载归一化参数，用于转换回摄氏度
    data_min = float(data_dict['metadata']['normalization_min'])
    data_max = float(data_dict['metadata']['normalization_max'])
    
    print(f"数据标准化范围: [{data_min:.3f}, {data_max:.3f}]°C")
    
    # 2. 创建数据集和数据加载器
    train_dataset = SSTDataset(train_X, train_y, ocean_mask)
    val_dataset = SSTDataset(val_X, val_y, ocean_mask)
    test_dataset = SSTDataset(test_X, test_y, ocean_mask)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"批次大小: {batch_size}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 3. 创建EKF-PredrNN模型 - 只使用一次降采样
    ekf_predrnn_model = EKFPredRNN(
        hidden_channels=32,
        state_dim=128,  
        num_layers=4,
        output_days=7
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in ekf_predrnn_model.parameters())
    trainable_params = sum(p.numel() for p in ekf_predrnn_model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 4. 训练模型
    ocean_mask_tensor = torch.FloatTensor(ocean_mask.astype(np.float32))
    
    train_losses, val_losses = train_model(
        model=ekf_predrnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        ocean_mask=ocean_mask_tensor,
        data_min=data_min,
        data_max=data_max,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        save_dir=save_dir
    )
    
    # 5. 加载最佳模型并测试
    print("\n加载最佳EKF-PredrNN模型进行测试...")
    model_path = os.path.join(save_dir, 'best_ekf_predrnn_7day_model.pth')
    ekf_predrnn_model.load_state_dict(torch.load(model_path))
    
    test_loss, test_preds, test_targets, test_metrics = evaluate_model(
        model=ekf_predrnn_model,
        test_loader=test_loader,
        ocean_mask=ocean_mask_tensor,
        data_min=data_min,
        data_max=data_max,
        device=device
    )
    
    # 6. 绘制预测趋势
    plot_prediction_trend(test_metrics, save_dir=save_dir)
    
    # 7. 可视化预测结果
    visualize_predictions(test_preds, test_targets, ocean_mask, data_min, data_max, num_samples=2, save_dir=save_dir)
    
    # 8. 保存结果
    results = {
        'test_loss': test_loss,
        'metrics': test_metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'data_params': {
            'min': data_min,
            'max': data_max
        }
    }
    
    results_path = os.path.join(save_dir, 'ekf_predrnn_7day_results.npy')
    np.save(results_path, results)
    print(f"\n结果已保存到 {results_path}")
    
    # 保存训练历史图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('EKF-PredrNN Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    history_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("EKF-PredrNN 7天预测训练完成!")
    print("=" * 60)
    print(f"最终测试损失: {test_loss:.6f}")
    print(f"平均RMSE: {test_metrics['average']['rmse_celsius']:.4f}°C")
    print(f"平均MAE: {test_metrics['average']['mae_celsius']:.4f}°C")
    print(f"平均R2: {test_metrics['average']['r2']:.6f}")
    
    return ekf_predrnn_model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EKF-PredrNN 海温预测模型')
    parser.add_argument('--data_file', type=str, default='./data/sst_final1.nc',
                        help='输入数据文件路径')
    parser.add_argument('--save_dir', type=str, default='./02results/ekf_predrnn',
                        help='结果保存目录')
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(data_file=args.data_file, save_dir=args.save_dir)