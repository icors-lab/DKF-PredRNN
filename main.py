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

# ==========================================
# 1. Global Settings & Random Seed
# ==========================================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return[convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return[convert_numpy_to_python(item) for item in obj]
    else:
        return obj


# ==========================================
# 2. Dataset & Data Processing
# ==========================================
class SSTDataset(Dataset):
    """SST Dataset for spatial-temporal sequence forecasting."""
    def __init__(self, X, y, ocean_mask):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.ocean_mask = torch.FloatTensor(ocean_mask.astype(np.float32))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data_dict, split='train', input_days=7, output_days=7):
    """Generate sliding window sequences for training/validation/testing."""
    sst_data = data_dict['sst_normalized']
    time_split = data_dict['time_split']
    
    split_mapping = {'train': 0, 'val': 1, 'test': 2}
    split_value = split_mapping[split]
    
    X, y = [],[]
    
    for i in range(len(sst_data) - input_days - output_days + 1):
        last_output_time = i + input_days + output_days - 1
        
        if time_split[last_output_time] == split_value:
            input_seq = sst_data[i:i + input_days]
            output_seq = sst_data[i + input_days:i + input_days + output_days]
            
            X.append(input_seq)
            y.append(output_seq)
    
    print(f"Generated {split} sequences: {len(X)}")
    return np.array(X), np.array(y)


def load_and_prepare_data(data_file):
    """Load preprocessed NetCDF data and prepare sequence arrays."""
    print(f"Loading preprocessed dataset: {data_file}")
    ds = xr.open_dataset(data_file)
    data_dict = {
        'sst_normalized': ds['sst_normalized'].values,
        'ocean_mask': ds['ocean_mask'].values.astype(bool),
        'time_split': ds['time_split'].values,
        'sst_original': ds['sst_original'].values,
        'metadata': dict(ds.attrs)
    }
    ds.close()
    
    print(f"Dataset loaded. Total shape: {data_dict['sst_normalized'].shape}")
    
    train_X, train_y = create_sequences(data_dict, 'train')
    val_X, val_y = create_sequences(data_dict, 'val')
    test_X, test_y = create_sequences(data_dict, 'test')
    
    # Expand channel dimension -> (N, T, C, H, W)
    train_X = np.expand_dims(train_X, axis=2)
    train_y = np.expand_dims(train_y, axis=2)
    val_X = np.expand_dims(val_X, axis=2)
    val_y = np.expand_dims(val_y, axis=2)
    test_X = np.expand_dims(test_X, axis=2)
    test_y = np.expand_dims(test_y, axis=2)
    
    print(f"Train shapes: X={train_X.shape}, y={train_y.shape}")
    print(f"Val shapes: X={val_X.shape}, y={val_y.shape}")
    print(f"Test shapes: X={test_X.shape}, y={test_y.shape}")
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y), data_dict['ocean_mask'], data_dict


# ==========================================
# 3. Loss Functions & Evaluation Metrics
# ==========================================
def ocean_masked_loss(pred, target, ocean_mask):
    """Compute MSE loss strictly within ocean regions."""
    batch_size, days, channels, height, width = pred.size()
    mask = ocean_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, days, channels, height, width)
    
    masked_pred = pred * mask
    masked_target = target * mask
    
    mse = nn.functional.mse_loss(masked_pred, masked_target, reduction='none')
    valid_pixels = mask.sum()
    
    if valid_pixels > 0:
        return mse.sum() / valid_pixels
    else:
        return mse.mean()


def calculate_metrics(preds, targets, ocean_mask, data_min, data_max):
    """Calculate RMSE, MAE, and R2 globally across ocean pixels."""
    output_days = preds.shape[1]
    metrics_by_day =[]
    ocean_mask = ocean_mask.astype(np.float32)
    
    for day in range(output_days):
        day_preds = preds[:, day, 0]
        day_targets = targets[:, day, 0]
        
        batch_size = day_preds.shape[0]
        batch_mask = np.repeat(np.expand_dims(ocean_mask, axis=0), batch_size, axis=0)
        
        masked_preds = day_preds * batch_mask
        masked_targets = day_targets * batch_mask
        valid_pixels = np.sum(batch_mask)
        
        squared_diff = (masked_preds - masked_targets) ** 2
        abs_diff = np.abs(masked_preds - masked_targets)
        
        mse = np.sum(squared_diff) / valid_pixels
        rmse = np.sqrt(mse)
        mae = np.sum(abs_diff) / valid_pixels
        
        masked_mean_target = np.sum(masked_targets) / valid_pixels
        ss_total = np.sum((masked_targets - masked_mean_target * batch_mask) ** 2)
        ss_residual = np.sum(squared_diff)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        rmse_celsius = rmse * (data_max - data_min)
        mae_celsius = mae * (data_max - data_min)
        
        metrics_by_day.append({
            'rmse': rmse, 'rmse_celsius': rmse_celsius,
            'mae': mae, 'mae_celsius': mae_celsius,
            'r2': r2
        })
    
    avg_metrics = {
        'rmse_celsius': np.mean([m['rmse_celsius'] for m in metrics_by_day]),
        'mae_celsius': np.mean([m['mae_celsius'] for m in metrics_by_day]),
        'r2': np.mean([m['r2'] for m in metrics_by_day])
    }
    
    return {'by_day': metrics_by_day, 'average': avg_metrics}


def visualize_predictions(preds, targets, ocean_mask, data_min, data_max, num_samples=2, save_dir="./results"):
    """
    Visualize spatial forecasting results by plotting Prediction, Ground Truth, 
    and Absolute Error maps with consistent color scales and land masking.
    """
    output_days = preds.shape[1]
    ocean_mask_bool = ocean_mask.astype(bool)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(output_days, 3, figsize=(15, 4.5 * output_days))
        
        for day in range(output_days):
            # De-normalize normalized SST values back to Celsius
            pred = preds[i, day, 0].copy()
            target = targets[i, day, 0].copy()
            diff = np.abs(pred - target)
            
            pred_celsius = pred * (data_max - data_min) + data_min
            target_celsius = target * (data_max - data_min) + data_min
            diff_err = diff * (data_max - data_min)
            
            # Apply land masking: set land pixels to NaN for transparent rendering
            pred_celsius[~ocean_mask_bool] = np.nan
            target_celsius[~ocean_mask_bool] = np.nan
            diff_err[~ocean_mask_bool] = np.nan
            
            # Sync colorbar scales for Prediction and Ground Truth columns
            vmin, vmax = np.nanmin(target_celsius), np.nanmax(target_celsius)
            vmax_err = np.nanmax(diff_err)
            
            # Subplot: Model Prediction
            im1 = axes[day, 0].imshow(pred_celsius, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            axes[day, 0].set_title(f'Day {day+1} - Prediction')
            axes[day, 0].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im1, ax=axes[day, 0], shrink=0.8, label='SST (°C)')
            axes[day, 0].axis('off')
            
            # Subplot: Ground Truth (Observation)
            im2 = axes[day, 1].imshow(target_celsius, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            axes[day, 1].set_title(f'Day {day+1} - Ground Truth')
            axes[day, 1].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im2, ax=axes[day, 1], shrink=0.8, label='SST (°C)')
            axes[day, 1].axis('off')
            
            # Subplot: Absolute Prediction Error
            im3 = axes[day, 2].imshow(diff_err, cmap='Reds', vmin=0, vmax=vmax_err)
            axes[day, 2].set_title(f'Day {day+1} - Absolute Error')
            axes[day, 2].contour(ocean_mask, levels=[0.5], colors='black', linewidths=0.5)
            plt.colorbar(im3, ax=axes[day, 2], shrink=0.8, label='Error (°C)')
            axes[day, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'dkf_predrnn_viz_sample_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def plot_prediction_trend(metrics, save_dir="./results"):
    """Plot forecasting metrics across lead times."""
    days = list(range(1, 8))
    rmse_values = [metrics['by_day'][i]['rmse_celsius'] for i in range(7)]
    mae_values = [metrics['by_day'][i]['mae_celsius'] for i in range(7)]
    r2_values = [metrics['by_day'][i]['r2'] for i in range(7)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(days, rmse_values, 'o-', color='blue', label='RMSE')
    ax1.plot(days, mae_values, 's-', color='red', label='MAE')
    ax1.set_xlabel('Lead Time (Days)')
    ax1.set_ylabel('Error (°C)')
    ax1.set_title('Forecasting Error over Lead Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(days, r2_values, 'd-', color='green', label='R²')
    ax2.set_xlabel('Lead Time (Days)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score over Lead Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_trend_metrics.png"), dpi=300)
    plt.close()


# ==========================================
# 4. Model Architecture: DKF-PredRNN++
# ==========================================
class DiffKalmanFilterLayer(nn.Module):
    """Differentiable Kalman Filter layer for Online State Correction."""
    def __init__(self, state_dim):
        super(DiffKalmanFilterLayer, self).__init__()
        self.state_dim = state_dim
        
        # State transition parameters (Low-rank decomposition)
        rank = min(state_dim, 24)
        self.F_low = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
        self.F_high = nn.Parameter(torch.randn(rank, state_dim) * 0.01)

        # Nonlinear dynamics compensation network
        self.nonlinear_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, state_dim),
            nn.Tanh()
        )
        self.nonlinear_weight = nn.Parameter(torch.tensor(0.1))

        # Noise covariance estimation
        self.log_Q_diag = nn.Parameter(torch.zeros(state_dim))
        self.log_R_diag = nn.Parameter(torch.zeros(state_dim))

    def forward(self, state, P, observation):
        batch_size = state.shape[0]
        device = state.device

        # Formulate state transition matrix with stability constraint
        F = torch.matmul(self.F_low, self.F_high) + torch.eye(self.state_dim, device=device)
        F = F.unsqueeze(0).expand(batch_size, -1, -1)

        Q = torch.diag_embed(torch.exp(self.log_Q_diag)).unsqueeze(0).expand(batch_size, -1, -1)
        R = torch.diag_embed(torch.exp(self.log_R_diag)).unsqueeze(0).expand(batch_size, -1, -1)

        # 1. Prediction Step
        state_pred = torch.bmm(F, state.unsqueeze(-1)).squeeze(-1)
        nonlinear_contrib = self.nonlinear_net(state)
        state_pred = state_pred + torch.sigmoid(self.nonlinear_weight) * nonlinear_contrib
        P_pred = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q

        # 2. Update Step (Observation matrix H is Identity)
        H = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        innovation = observation - state_pred
        
        S = torch.bmm(torch.bmm(H, P_pred), H.transpose(1, 2)) + R
        S_reg = S + 1e-5 * torch.eye(self.state_dim, device=device).unsqueeze(0)
        S_inv = torch.inverse(S_reg)
        K = torch.bmm(torch.bmm(P_pred, H.transpose(1, 2)), S_inv)

        state_updated = state_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)

        # Joseph form covariance update for numerical stability
        I = torch.eye(self.state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        term1 = I - torch.bmm(K, H)
        P_updated = torch.bmm(torch.bmm(term1, P_pred), term1.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K, R), K.transpose(1, 2))

        return state_updated, P_updated, K


class DKFSTLSTMCell(nn.Module):
    """Spatiotemporal LSTM Cell integrated with Differentiable Kalman Filter."""
    def __init__(self, hidden_channels, state_dim, kernel_size=3, img_size=(32, 32)):
        super(DKFSTLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.state_dim = state_dim
        padding = kernel_size // 2

        self.conv_gates = nn.Conv2d(hidden_channels * 3, hidden_channels * 4, kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        # State Encoder
        self.state_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, state_dim),
            nn.LayerNorm(state_dim)
        )

        # State Decoder
        self.state_decoder = nn.Sequential(
            nn.Linear(state_dim, 32 * 16 * 16),
            nn.LeakyReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.Upsample(size=img_size, mode='bilinear', align_corners=False),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
            nn.LeakyReLU()
        )

        self.kf_layer = DiffKalmanFilterLayer(state_dim)
        self.kf_alpha = nn.Parameter(torch.tensor(0.3))
        self.res_weight = nn.Parameter(torch.tensor(0.1))

        self.input_conv_1 = nn.Conv2d(1, hidden_channels, kernel_size=1)
        self.input_conv_3 = nn.Conv2d(3, hidden_channels, kernel_size=1)

    def forward(self, x, h, c, m, kf_state=None, kf_cov=None):
        if x.size(1) != self.hidden_channels:
            if x.size(1) == 1:
                x = self.input_conv_1(x)
            elif x.size(1) == 3:
                x = self.input_conv_3(x)
            else:
                input_conv = nn.Conv2d(x.size(1), self.hidden_channels, kernel_size=1).to(x.device)
                x = input_conv(x)

        _, _, H_x, W_x = x.shape
        _, _, H_h, W_h = h.shape

        if H_x != H_h or W_x != W_h:
            h = F.interpolate(h, size=(H_x, W_x), mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=(H_x, W_x), mode='bilinear', align_corners=False)
            m = F.interpolate(m, size=(H_x, W_x), mode='bilinear', align_corners=False)

        h_orig = h

        # 1. Base ST-LSTM Update
        combined = torch.cat([x, h, m], dim=1)
        gates = self.conv_gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        m_next = torch.sigmoid(self.conv_m(c_next)) * torch.tanh(c_next)
        h_lstm = o * torch.tanh(c_next)

        # 2. Kalman Filter State Correction
        batch_size = h.shape[0]
        if kf_state is None:
            kf_state = torch.zeros(batch_size, self.state_dim, device=h.device)
        if kf_cov is None:
            kf_cov = torch.eye(self.state_dim).unsqueeze(0).repeat(batch_size, 1, 1).to(h.device) * 0.1

        encoded_h = self.state_encoder(h_lstm)
        kf_state_next, kf_cov_next, _ = self.kf_layer(kf_state, kf_cov, encoded_h)
        decoded_state = self.state_decoder(kf_state_next)

        if decoded_state.shape[2:] != h_lstm.shape[2:]:
            decoded_state = F.interpolate(decoded_state, size=h_lstm.shape[2:], mode='bilinear')

        # 3. Adaptive Fusion & Residual Connection
        alpha = torch.sigmoid(self.kf_alpha)
        h_next = h_lstm * (1 - alpha) + decoded_state * alpha
        
        res_scale = torch.sigmoid(self.res_weight)
        h_next = h_next + res_scale * h_orig

        return h_next, c_next, m_next, kf_state_next, kf_cov_next


class EKFPredRNN(nn.Module):
    """Main Architecture: DKF-PredRNN++."""
    def __init__(self, hidden_channels=64, state_dim=32, num_layers=3, output_days=7):
        super(EKFPredRNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.output_days = output_days
        self.img_size = (32, 32)

        # Single spatial downsampling (64x64 -> 32x32)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.st_lstm_cells = nn.ModuleList([
            DKFSTLSTMCell(hidden_channels, state_dim, img_size=self.img_size)
            for _ in range(num_layers)
        ])

        self.memory_flow_cells = nn.ModuleList([
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1)
            for _ in range(num_layers - 1)
        ])

        self.time_embedding = nn.Parameter(torch.zeros(1, 14, hidden_channels, 32, 32))
        nn.init.xavier_uniform_(self.time_embedding)

        # Decoder to original resolution (32x32 -> 64x64)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_sequence(self, x):
        B, T, C, H, W = x.shape
        device = x.device
        H_encoded, W_encoded = self.img_size

        h_list =[torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device) for _ in range(self.num_layers)]
        c_list =[torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device) for _ in range(self.num_layers)]
        m_list =[torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device) for _ in range(self.num_layers)]
        
        kf_states, kf_covs = [None] * self.num_layers, [None] * self.num_layers
        memory_flows =[torch.zeros(B, self.hidden_channels, H_encoded, W_encoded, device=device) for _ in range(self.num_layers - 1)]

        for t in range(T):
            x_t = self.encoder(x[:, t])

            for l in range(self.num_layers):
                if t < self.time_embedding.size(1):
                    x_t = x_t + self.time_embedding[:, t]

                if l > 0 and t > 0:
                    flow = self.memory_flow_cells[l - 1](torch.cat([x_t, memory_flows[l - 1]], dim=1))
                    memory_flows[l - 1] = flow
                    x_t = x_t + 0.2 * flow

                h_next, c_next, m_next, kf_state_next, kf_cov_next = self.st_lstm_cells[l](
                    x_t, h_list[l], c_list[l], m_list[l], kf_states[l], kf_covs[l]
                )

                h_list[l], c_list[l], m_list[l] = h_next, c_next, m_next
                kf_states[l], kf_covs[l] = kf_state_next, kf_cov_next
                x_t = h_next

        return h_list, c_list, m_list, kf_states, kf_covs, memory_flows

    def _decode_step(self, h_list, c_list, m_list, kf_states, kf_covs, memory_flows, decoder_input, t):
        x_t = self.encoder(decoder_input)
        
        for l in range(self.num_layers):
            if t + 7 < self.time_embedding.size(1):
                x_t = x_t + self.time_embedding[:, t + 7]
            
            if l > 0:
                flow = self.memory_flow_cells[l - 1](torch.cat([x_t, memory_flows[l - 1]], dim=1))
                memory_flows[l - 1] = flow
                x_t = x_t + 0.2 * flow
            
            h_next, c_next, m_next, kf_state_next, kf_cov_next = self.st_lstm_cells[l](
                x_t, h_list[l], c_list[l], m_list[l], kf_states[l], kf_covs[l]
            )
            
            h_list[l], c_list[l], m_list[l] = h_next, c_next, m_next
            kf_states[l], kf_covs[l] = kf_state_next, kf_cov_next
            x_t = h_next
        
        return self.decoder(h_list[-1]), h_list, c_list, m_list, kf_states, kf_covs, memory_flows

    def forward(self, x, target_tensor=None, teacher_forcing_ratio=0.0):
        h_list, c_list, m_list, kf_states, kf_covs, memory_flows = self._encode_sequence(x)
        outputs =[]
        decoder_input = x[:, -1]
        
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        for t in range(self.output_days):
            output, h_list, c_list, m_list, kf_states, kf_covs, memory_flows = self._decode_step(
                h_list, c_list, m_list, kf_states, kf_covs, memory_flows, decoder_input, t
            )
            outputs.append(output)
            
            if use_teacher_forcing and target_tensor is not None and t < self.output_days - 1:
                decoder_input = target_tensor[:, t]
            else:
                decoder_input = output
                
        return torch.stack(outputs, dim=1)


# ==========================================
# 5. Training & Evaluation Pipeline
# ==========================================
def evaluate_model(model, test_loader, ocean_mask, data_min, data_max, device='cuda'):
    model.eval()
    model = model.to(device)
    ocean_mask = ocean_mask.to(device)
    
    test_loss, test_batches = 0.0, 0
    all_preds, all_targets = [],[]
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing Inference'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x, teacher_forcing_ratio=0.0)
            
            loss = ocean_masked_loss(pred, batch_y, ocean_mask)
            test_loss += loss.item()
            test_batches += 1
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_preds, all_targets, ocean_mask.cpu().numpy(), data_min, data_max)
    
    print(f"\n[{'Test Results':^20}]")
    print(f"Overall MSE: {test_loss / test_batches:.6f}\n")
    
    # === 新加回来的每天详细数据打印 ===
    print("Daily Breakdown:")
    print("-" * 50)
    for day in range(all_preds.shape[1]):
        day_metrics = metrics['by_day'][day]
        print(f"  Day {day+1} | RMSE: {day_metrics['rmse_celsius']:.4f}°C | "
              f"MAE: {day_metrics['mae_celsius']:.4f}°C | "
              f"R2: {day_metrics['r2']:.4f}")
    print("-" * 50)
    # ===============================

    print(f"Average RMSE: {metrics['average']['rmse_celsius']:.4f}°C")
    print(f"Average MAE:  {metrics['average']['mae_celsius']:.4f}°C")
    print(f"Average R2:   {metrics['average']['r2']:.4f}\n")
    
    return test_loss / test_batches, all_preds, all_targets, metrics


def train_model(model, train_loader, val_loader, ocean_mask, data_min, data_max, 
                num_epochs=50, learning_rate=0.0001, device='cuda', save_dir="./results"):
    
    if num_epochs == 0:
        print("num_epochs set to 0. Skipping training phase...")
        return [],[]

    model = model.to(device)
    ocean_mask = ocean_mask.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scaler = amp.GradScaler()
    
    train_losses, val_losses = [],[]
    best_val_loss = float('inf')
    early_stop_patience = 20
    no_improve_epochs = 0
    
    print("\nStarting Training: DKF-PredRNN++...")
    for epoch in range(num_epochs):
        t0 = time.time()
        teacher_forcing_ratio = max(0.0, 1.0 - epoch * 0.003)
        
        # Training
        model.train()
        train_loss, train_batches = 0.0, 0
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(batch_x, target_tensor=batch_y, teacher_forcing_ratio=teacher_forcing_ratio)
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
        
        # Validation
        model.eval()
        val_loss, val_batches = 0.0, 0
        val_preds, val_targets = [],[]
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}[Val]'):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x, teacher_forcing_ratio=0.0)
                loss = ocean_masked_loss(pred, batch_y, ocean_mask)
                
                val_loss += loss.item()
                val_batches += 1
                val_preds.append(pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
                
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        metrics = calculate_metrics(np.concatenate(val_preds), np.concatenate(val_targets), 
                                    ocean_mask.cpu().numpy(), data_min, data_max)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time.time()-t0:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | TF Ratio: {teacher_forcing_ratio:.3f}")
        print(f"  Val RMSE: {metrics['average']['rmse_celsius']:.4f}°C")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_ekf_predrnn_7day_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved best model to {model_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
                break
                
    return train_losses, val_losses


# ==========================================
# 6. Main Function
# ==========================================
def main(args):
    print("=" * 60)
    print("DKF-PredRNN++: SST Sequence Forecasting Model (7-day input -> 7-day output)")
    print("=" * 60)
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device configuration: {device}")
    
    # Load and process data
    data_res = load_and_prepare_data(args.data_file)
    (train_X, train_y), (val_X, val_y), (test_X, test_y), ocean_mask, data_dict = data_res
    
    data_min = float(data_dict['metadata']['normalization_min'])
    data_max = float(data_dict['metadata']['normalization_max'])
    
    # Build DataLoaders
    batch_size = 16
    train_loader = DataLoader(SSTDataset(train_X, train_y, ocean_mask), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(SSTDataset(val_X, val_y, ocean_mask), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(SSTDataset(test_X, test_y, ocean_mask), batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize Model
    model = EKFPredRNN(hidden_channels=32, state_dim=128, num_layers=4, output_days=7)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model successfully initialized. Trainable parameters: {trainable_params:,}")
    
    # Train
    ocean_mask_tensor = torch.FloatTensor(ocean_mask.astype(np.float32))
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        ocean_mask=ocean_mask_tensor, data_min=data_min, data_max=data_max,
        num_epochs=args.epochs, learning_rate=0.001, device=device, save_dir=args.save_dir
    )
    
    # Evaluate
    print("\nLoading the best DKF-PredRNN++ model for final evaluation...")
    model_path = os.path.join(args.save_dir, 'best_ekf_predrnn_7day_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    else:
        print(f"Warning: Best model weights not found at {model_path}. Evaluating current state.")
        
    test_loss, test_preds, test_targets, test_metrics = evaluate_model(
        model=model, test_loader=test_loader, ocean_mask=ocean_mask_tensor,
        data_min=data_min, data_max=data_max, device=device
    )
    
    # Visualizations
    plot_prediction_trend(test_metrics, save_dir=args.save_dir)
    visualize_predictions(test_preds, test_targets, ocean_mask, data_min, data_max, num_samples=2, save_dir=args.save_dir)
    
    if args.epochs > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DKF-PredRNN++ Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n" + "=" * 60)
    print("Process Finished Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DKF-PredRNN++ SST Forecasting Model')
    parser.add_argument('--data_file', type=str, default='./data/sst_final1.nc', help='Path to preprocessed NetCDF data file')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results and model weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (set to 0 for inference only)')
    args = parser.parse_args()
    
    main(args)