import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 1. 超参数与路径设置 =================
HIST_LEN = 30
FUTURE_LEN = 50
IN_CHANNELS = 10
MAX_NODES = 8

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 3e-4

DATA_DIR = os.path.expanduser("~/Desktop/processed_trajectory_following")
DEVICE = torch.device("cpu")
print(f"Started on {DEVICE}")

# ================= 2. 核心网络模型 =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, h, adj):
        Wh = self.W(h)
        N = Wh.size(1)
        a_input = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        a_input = torch.cat([a_input, a_input.permute(0, 2, 1, 3)], dim=-1)
        e = F.leaky_relu(self.a(a_input).squeeze(-1))
        e = e.masked_fill(adj == 0.0, -1e9)
        attention = F.softmax(e, dim=-1)
        return torch.matmul(attention, Wh)


class TrajectoryGNNTransformer(nn.Module):
    def __init__(
        self,
        in_channels=10,
        embed_dim=128,
        lstm_hidden=128,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
        future_len=50,
        num_modes=6,
    ):
        super().__init__()
        self.future_len = future_len
        self.num_modes = num_modes

        # 时序编码器
        self.lstm = nn.LSTM(in_channels, lstm_hidden, batch_first=True)

        # 空间交互：只在最后一帧做一次 GNN，速度提升 30x
        self.gnn = GraphAttentionLayer(lstm_hidden, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        # 节点维度上的 Transformer（不再是时间维度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器：输出 num_modes 条归一化 delta 轨迹
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_modes * future_len * 2),
        )

    def forward(self, history, adj):
        # history: [B, T, N, C]
        # adj:     [B, T, N, N]
        B, T, N, C = history.shape

        # 1. 时序编码：每辆车独立跑 LSTM
        hist_flat = history.permute(0, 2, 1, 3).contiguous().view(B * N, T, C)
        _, (h_n, _) = self.lstm(hist_flat)          # h_n: [1, B*N, H]
        h_last = h_n[-1].view(B, N, -1)             # [B, N, H]

        # 2. 空间交互：用最后一帧邻接矩阵做一次 GNN
        x_gnn = self.gnn(h_last, adj[:, -1])        # [B, N, D]

        # 3. 节点维度 Transformer（建模车辆间的全局关系）
        x_trans = self.pos_enc(x_gnn)               # [B, N, D]
        x_trans = self.transformer(x_trans)          # [B, N, D]

        # 4. Masked Global Pooling（只聚合真实存在的邻居节点）
        valid_mask = (history[:, -1, :, 9] > 0.5).float().unsqueeze(-1)  # [B, N, 1]
        x_ego = x_trans[:, 0]                       # [B, D]
        x_global = (x_trans * valid_mask).sum(dim=1) / (
            valid_mask.sum(dim=1) + 1e-6
        )                                            # [B, D]

        x_main = torch.cat([x_ego, x_global], dim=-1)  # [B, D*2]

        # 5. 解码：输出 num_modes 条归一化 delta 轨迹
        pred = self.decoder(x_main).view(B, self.num_modes, self.future_len, 2)
        return pred  # [B, num_modes, future_len, 2]


# ================= 3. 数据集 =================
class NGSIMDataset(Dataset):
    def __init__(self, data_folder):
        self.file_paths = glob.glob(os.path.join(data_folder, "*.npy"))
        print(f"Loaded {len(self.file_paths)} samples from {data_folder}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        history = torch.tensor(data["history"], dtype=torch.float32)        # [T, N, C]
        history_graph = torch.tensor(data["history_graph"], dtype=torch.float32)
        future = torch.tensor(data["future"], dtype=torch.float32)          # [50, 2]

        # 数据增强：50% 概率左右镜像
        if torch.rand(1).item() > 0.5:
            history[:, :, 0] *= -1.0    # X 位移取反
            future[:, 0] *= -1.0
            dist_l_temp = history[:, :, 4].clone()
            history[:, :, 4] = history[:, :, 5]
            history[:, :, 5] = dist_l_temp
            history[:, :, 8] *= -1.0    # cos_h 不变，sin_h 取反

        return history, history_graph, future


# ================= 4. Z-Score 统计量计算 =================
def compute_dataset_statistics(dataloader):
    """
    计算两类统计量：
    1. feat_mean / feat_std：用于输入特征的 Z-Score 归一化
    2. delta_std：用于输出 delta 的归一化（让模型学习标准化的位移）

    修复点：
    - delta 使用正确的帧间差分（不用零填充 prepend，避免第一步误差虚大）
    - X 和 Y 使用统一的空间标准差，保证各向同性
    """
    print("\n计算全局统计量...")
    all_features, all_deltas = [], []

    for history, _, future in dataloader:
        valid_mask = history[..., 9] > 0.5
        all_features.append(history[valid_mask])

        # 修复：正确的帧间差分
        # future[:, 0] 是 t=HIST 时刻相对锚点的位移，不是 delta
        # 正确做法：计算相邻帧之间的位移差
        delta = future[:, 1:] - future[:, :-1]  # [B, 49, 2]
        # 第一步 delta 用第一个位移本身近似（t=0 时在锚点，即 (0,0)）
        first_delta = future[:, :1]              # [B, 1, 2]
        delta = torch.cat([first_delta, delta], dim=1)  # [B, 50, 2]
        all_deltas.append(delta)

    all_features = torch.cat(all_features, dim=0)
    all_deltas = torch.cat(all_deltas, dim=0)   # [Total, 50, 2]

    # 静态特征统计
    feat_mean = all_features.mean(dim=0)
    feat_std = all_features.std(dim=0)

    # X 和 Y 使用统一标准差，保证各向同性
    spatial_std = torch.sqrt((feat_std[0] ** 2 + feat_std[1] ** 2) / 2.0)
    feat_std[0] = spatial_std
    feat_std[1] = spatial_std
    feat_std[feat_std < 1e-5] = 1.0

    # 特征 6~9（lane_type, sin_h, cos_h, mask）已在 [-1,1] 或 {0,1}，不需要归一化
    feat_mean[6:] = 0.0
    feat_std[6:] = 1.0

    # Delta 统计：X 和 Y 同样使用统一标准差
    delta_std_raw = all_deltas.std(dim=(0, 1))  # [2]
    delta_spatial_std = torch.sqrt(
        (delta_std_raw[0] ** 2 + delta_std_raw[1] ** 2) / 2.0
    )
    delta_std = torch.tensor([delta_spatial_std, delta_spatial_std])

    print(
        f"空间标准差: {spatial_std.item():.3f} m | "
        f"Delta 标准差: {delta_spatial_std.item():.4f} m/frame"
    )
    return feat_mean.to(DEVICE), feat_std.to(DEVICE), delta_std.to(DEVICE)


# ================= 5. 指标计算 =================
def calculate_min_metrics(pred_traj, future_traj):
    """
    pred_traj:   [B, num_modes, 50, 2]，物理米，相对锚点
    future_traj: [B, 50, 2]，物理米，相对锚点

    返回 minADE 和 minFDE（每个样本取最优模态，再求均值）
    """
    future_traj = future_traj.unsqueeze(1)                   # [B, 1, 50, 2]
    l2 = torch.norm(pred_traj - future_traj, dim=-1)         # [B, num_modes, 50]

    ade_per_mode = l2.mean(dim=2)                            # [B, num_modes]
    min_ade = ade_per_mode.min(dim=1)[0].mean().item()

    fde_per_mode = l2[:, :, -1]                              # [B, num_modes]
    min_fde = fde_per_mode.min(dim=1)[0].mean().item()

    return min_ade, min_fde


# ================= 6. 主训练流程 =================
def main():
    train_dataset = NGSIMDataset(os.path.join(DATA_DIR, "train"))
    test_dataset = NGSIMDataset(os.path.join(DATA_DIR, "test"))

    num_workers = 0  # CPU 环境用 0，有 GPU 改为 4
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    # 计算归一化统计量
    feat_mean, feat_std, delta_std = compute_dataset_statistics(train_loader)

    model = TrajectoryGNNTransformer(in_channels=IN_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    best_ade = float('inf')

    for epoch in range(EPOCHS):
        # -------- 训练 --------
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for history, history_graph, future in pbar:
            history = history.to(DEVICE)
            history_graph = history_graph.to(DEVICE)
            future = future.to(DEVICE)

            # 输入特征 Z-Score 归一化
            history_norm = (history - feat_mean) / feat_std

            optimizer.zero_grad()

            # 模型输出：归一化 delta [B, 6, 50, 2]
            pred_delta_norm = model(history_norm, history_graph)

            # GT delta 归一化
            # 修复：与统计量计算保持一致的 delta 定义
            first_delta = future[:, :1]                            # [B, 1, 2]
            rest_delta = future[:, 1:] - future[:, :-1]           # [B, 49, 2]
            future_delta = torch.cat([first_delta, rest_delta], dim=1)  # [B, 50, 2]
            future_delta_norm = (future_delta / delta_std).unsqueeze(1)  # [B, 1, 50, 2]

            # Winner-Takes-All Loss
            # 计算每条模态轨迹的 L1 总误差
            all_losses = torch.sum(
                torch.abs(pred_delta_norm - future_delta_norm), dim=(2, 3)
            )  # [B, 6]

            # epsilon-greedy WTA：90% 只优化最优模态，10% 全部模态都学
            if torch.rand(1).item() > 0.1:
                # 硬 WTA：只给最优模态反传梯度
                best_idx = all_losses.argmin(dim=1)               # [B]
                loss = all_losses.gather(
                    1, best_idx.unsqueeze(1)
                ).squeeze(1).mean()
            else:
                # 探索：所有模态都优化，防止模态梯度饥饿
                loss = all_losses.mean()

            loss = loss / (FUTURE_LEN * 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.5f}",
                "LR": f"{scheduler.get_last_lr()[0]:.5f}"
            })

        scheduler.step()

        # -------- 评估 --------
        model.eval()
        total_ade, total_fde = 0.0, 0.0

        with torch.no_grad():
            for history, history_graph, future in tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eval]"
            ):
                history = history.to(DEVICE)
                history_graph = history_graph.to(DEVICE)
                future = future.to(DEVICE)

                history_norm = (history - feat_mean) / feat_std

                # 模型预测：归一化 delta
                pred_delta_norm = model(history_norm, history_graph)  # [B, 6, 50, 2]

                # 反归一化：delta → 物理米
                pred_delta = pred_delta_norm * delta_std.view(1, 1, 1, 2)  # [B, 6, 50, 2]

                # 修复：cumsum 得到相对锚点的累积位移（物理米）
                # future 本身就是相对锚点的累积位移，不需要加任何偏移
                pred_traj = pred_delta.cumsum(dim=2)  # [B, 6, 50, 2]

                # 计算 minADE / minFDE
                ade, fde = calculate_min_metrics(pred_traj, future)
                total_ade += ade
                total_fde += fde

        mean_ade = total_ade / len(test_loader)
        mean_fde = total_fde / len(test_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss/len(train_loader):.5f} | "
            f"minADE: {mean_ade:.4f} m | "
            f"minFDE: {mean_fde:.4f} m"
        )

        if mean_ade < best_ade:
            best_ade = mean_ade
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> New best saved (minADE: {best_ade:.4f} m)")


if __name__ == "__main__":
    main()