import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ==========================================
# 0. 路径与超参数（与 train_fixed.py 完全一致）
# ==========================================
DATA_DIR   = os.path.expanduser("~/Desktop/processed_trajectory_merging")
TEST_DIR   = os.path.join(DATA_DIR, "test")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
MODEL_PATH = "best_model.pth"
SAVE_DIR   = "trajectory_plots_v3"
os.makedirs(SAVE_DIR, exist_ok=True)

IN_CHANNELS = 10
HIST_LEN    = 30
FUTURE_LEN  = 50
MAX_NODES   = 8
NUM_MODES   = 6
EMBED_DIM   = 128
LSTM_HIDDEN = 128
NUM_HEADS   = 8
FF_DIM      = 512
NUM_LAYERS  = 4

NUM_PLOTS = 50          # 总出图数
DEVICE    = torch.device("cpu")

# 美学配置（复刻旧代码风格）
BG_COLOR   = '#F5F2EB'  # 羊皮纸
GRID_COLOR = '#E2DCD0'
HIST_COLOR = '#777777'  # 历史轨迹灰
GT_COLOR   = '#222222'  # 真值黑
EGO_COLOR  = '#D32F2F'  # 主车红
OTH_COLOR  = '#909090'  # 邻居灰

# ==========================================
# 1. 模型定义（与 train_fixed.py 完全相同）
# ==========================================
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
        attn = F.softmax(e, dim=-1)
        return torch.matmul(attn, Wh), attn   # 同时返回注意力权重


class TrajectoryGNNTransformer(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128, lstm_hidden=128,
                 num_heads=8, ff_dim=512, num_layers=4,
                 future_len=50, num_modes=6):
        super().__init__()
        self.future_len = future_len
        self.num_modes  = num_modes
        self.lstm    = nn.LSTM(in_channels, lstm_hidden, batch_first=True)
        self.gnn     = GraphAttentionLayer(lstm_hidden, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, num_modes * future_len * 2),
        )

    def forward(self, history, adj, return_attn=False):
        B, T, N, C = history.shape
        hist_flat = history.permute(0, 2, 1, 3).contiguous().view(B * N, T, C)
        _, (h_n, _) = self.lstm(hist_flat)
        h_last = h_n[-1].view(B, N, -1)

        x_gnn, attn = self.gnn(h_last, adj[:, -1])   # attn: [B, N, N]
        x_trans = self.transformer(self.pos_enc(x_gnn))

        valid_mask = (history[:, -1, :, 9] > 0.5).float().unsqueeze(-1)
        x_ego    = x_trans[:, 0]
        x_global = (x_trans * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-6)
        x_main   = torch.cat([x_ego, x_global], dim=-1)

        pred = self.decoder(x_main).view(B, self.num_modes, self.future_len, 2)
        if return_attn:
            return pred, attn
        return pred


# ==========================================
# 2. 数据集
# ==========================================
class NGSIMDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.npy")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx], allow_pickle=True).item()
        return (
            torch.tensor(d["history"],       dtype=torch.float32),
            torch.tensor(d["history_graph"], dtype=torch.float32),
            torch.tensor(d["future"],        dtype=torch.float32),
        )


# ==========================================
# 3. 归一化统计量（与训练完全一致）
# ==========================================
def compute_statistics(train_dir, batch_size=256):
    print("计算归一化统计量...")
    loader = DataLoader(NGSIMDataset(train_dir),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    all_feat, all_delta = [], []
    for history, _, future in loader:
        all_feat.append(history[history[..., 9] > 0.5])
        fd = torch.cat([future[:, :1], future[:, 1:] - future[:, :-1]], dim=1)
        all_delta.append(fd)

    all_feat  = torch.cat(all_feat,  dim=0)
    all_delta = torch.cat(all_delta, dim=0)

    feat_mean = all_feat.mean(0)
    feat_std  = all_feat.std(0)
    sp = torch.sqrt((feat_std[0] ** 2 + feat_std[1] ** 2) / 2.0)
    feat_std[0] = feat_std[1] = sp
    feat_std[feat_std < 1e-5] = 1.0
    feat_mean[6:] = 0.0
    feat_std[6:]  = 1.0

    ds  = all_delta.std(dim=(0, 1))
    dsp = torch.sqrt((ds[0] ** 2 + ds[1] ** 2) / 2.0)
    delta_std = torch.tensor([dsp, dsp])
    print(f"  spatial std={sp.item():.3f} m  delta std={dsp.item():.4f} m/frame")
    return feat_mean, feat_std, delta_std


# ==========================================
# 4. 推理（返回最优模态轨迹 + 注意力权重）
# ==========================================
@torch.no_grad()
def predict_best(model, history, history_graph, feat_mean, feat_std, delta_std, future):
    """
    返回:
        best_traj [FUTURE_LEN, 2]  最优模态物理轨迹（相对锚点，米）
        ade, fde  float
        attn      [N, N] ego 行的注意力权重
    """
    hn = (history.unsqueeze(0) - feat_mean) / feat_std
    pred_dn, attn = model(hn, history_graph.unsqueeze(0), return_attn=True)
    pred_d   = pred_dn * delta_std.view(1, 1, 1, 2)
    pred_traj = pred_d.cumsum(dim=2).squeeze(0).cpu().numpy()   # [6, 50, 2]

    fut_np = future.numpy()   # [50, 2]
    l2 = np.linalg.norm(pred_traj - fut_np[np.newaxis], axis=-1)  # [6, 50]
    ade_per = l2.mean(1)
    best    = ade_per.argmin()
    ade_val = ade_per[best]
    fde_val = l2[best, -1]

    # ego→其他节点的注意力：取最后一层 GNN，ego 行
    attn_ego = attn[0, 0, :].cpu().numpy()   # [N]

    return pred_traj[best], ade_val, fde_val, attn_ego


# ==========================================
# 5. 坐标变换（横向布局）
# ==========================================
def to_plot(xy):
    """[N,2] (x=lateral, y=longitudinal) → plot (横轴=y, 纵轴=-x)"""
    return np.stack([xy[:, 1], -xy[:, 0]], axis=1)


def yaw_from_traj(traj):
    if len(traj) < 2:
        return np.zeros(1)
    dx = np.diff(traj[:, 0], append=traj[-1, 0])
    dy = np.diff(traj[:, 1], append=traj[-1, 1])
    return np.arctan2(dy, dx)


# ==========================================
# 6. 绘图工具（复刻旧代码）
# ==========================================
def draw_heatmap(ax, cx, cy, weight, max_w):
    """高斯热力图（注意力可视化）"""
    if max_w < 1e-9 or weight / max_w <= 0.05:
        return
    rw = weight / max_w
    x  = np.linspace(cx - 18, cx + 18, 100)
    y  = np.linspace(cy - 10, cy + 10, 100)
    X, Y = np.meshgrid(x, y)
    sx, sy = 8.0 + 4.0 * rw, 3.0 + 2.0 * rw
    Z  = np.exp(-((X - cx) ** 2 / sx + (Y - cy) ** 2 / sy)) * rw
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'TR', [(0.75, 0.10, 0.10, 0.0), (0.75, 0.10, 0.10, 0.95)]
    )
    cf = ax.contourf(X, Y, Z, levels=60, cmap=cmap,
                     vmin=0, vmax=1.0, zorder=2)
    if hasattr(cf, 'collections'):
        for c in cf.collections:
            c.set_edgecolor("face")
    else:
        cf.set_edgecolor("face")


def draw_car(ax, x, y, yaw, color='red', scale=1.1, alpha=1.0, zorder=10):
    """3D 俯视车辆模型（复刻旧代码）"""
    L, W = 4.8 * scale, 2.0 * scale
    tr = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData

    # 阴影
    ax.add_patch(FancyBboxPatch(
        (-L / 2 - 0.3, -W / 2 - 0.3), L, W,
        boxstyle="round,pad=0,rounding_size=0.4",
        ec='none', fc='black', alpha=0.25 * alpha,
        transform=tr, zorder=zorder - 2
    ))
    # 轮胎
    wl, ww = L * 0.18, W * 0.22
    for wx, wy in [
        ( L / 2 - wl,  W / 2 - ww / 2),
        ( L / 2 - wl, -W / 2 - ww / 2),
        (-L / 2,       W / 2 - ww / 2),
        (-L / 2,      -W / 2 - ww / 2),
    ]:
        ax.add_patch(Rectangle(
            (wx, wy), wl, ww,
            color='#222222', transform=tr, zorder=zorder - 1
        ))
    # 车身
    ax.add_patch(FancyBboxPatch(
        (-L / 2, -W / 2), L, W,
        boxstyle="round,pad=0,rounding_size=0.4",
        ec='#333333', lw=1.2, fc=color, alpha=alpha,
        transform=tr, zorder=zorder
    ))
    # 前窗
    ax.add_patch(FancyBboxPatch(
        (L * 0.08, -W * 0.35), L * 0.22, W * 0.7,
        boxstyle="round,pad=0,rounding_size=0.08",
        ec='black', lw=0.8, fc='#2A2A2A', alpha=alpha * 0.9,
        transform=tr, zorder=zorder + 1
    ))
    # 后窗
    ax.add_patch(FancyBboxPatch(
        (-L * 0.35, -W * 0.35), L * 0.15, W * 0.7,
        boxstyle="round,pad=0,rounding_size=0.05",
        ec='black', lw=0.8, fc='#2A2A2A', alpha=alpha * 0.9,
        transform=tr, zorder=zorder + 1
    ))


def setup_axis(ax, cx, cy, x_range=35, y_range=8):
    """坐标轴基础设置：范围、背景、车道线"""
    ax.set_xlim(cx - x_range, cx + x_range)
    ax.set_ylim(cy - y_range, cy + y_range)
    ax.set_aspect('equal')
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, linestyle='-', lw=0.6, color=GRID_COLOR, zorder=0)

    # 虚线车道线
    base = round(cy / 3.75) * 3.75
    for off in [-7.5, -3.75, 0, 3.75, 7.5]:
        ax.axhline(base + off, color='#666666', linestyle='--',
                   lw=1.5, dashes=(5, 5), zorder=1)

    ax.set_xlabel('y / m', fontsize=16,
                  fontfamily='Times New Roman', labelpad=10)
    ax.set_ylabel('x / m', fontsize=16,
                  fontfamily='Times New Roman', labelpad=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(14)
        lbl.set_fontname('Times New Roman')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.2)


# ==========================================
# 7. 两种子图渲染
# ==========================================
def render_pred(ax, hist_np, fut_np, pred_np, others_np):
    """
    预测图：历史灰线 + 真值黑线 + jet渐变预测线 + 3D车模
    hist_np  [T, 2]       相对锚点，物理米
    fut_np   [50, 2]
    pred_np  [50, 2]
    others_np list of [Ti, 2]
    """
    # 坐标变换
    h  = to_plot(hist_np)
    f  = to_plot(fut_np)
    p  = to_plot(pred_np)
    cx, cy = h[-1, 0], h[-1, 1]

    setup_axis(ax, cx, cy)

    # Neighbors: draw car marker only at last frame, no trajectory lines
    # (node identity is unstable across frames - drawing lines causes artifacts)
    for ov in others_np:
        if len(ov) < 1:
            continue
        ov_p = to_plot(ov)
        last = ov_p[-1]
        if not (cx - 35 <= float(last[0]) <= cx + 35):
            continue
        if len(ov_p) >= 2:
            yaw = math.atan2(float(ov_p[-1,1] - ov_p[-2,1]),
                             float(ov_p[-1,0] - ov_p[-2,0]))
        else:
            yaw = 0.0
        draw_car(ax, float(last[0]), float(last[1]), yaw,
                 color=OTH_COLOR, scale=1.1, zorder=3)

    # 历史轨迹（灰）
    ax.plot(h[:, 0], h[:, 1],
            color=HIST_COLOR, lw=3.5, zorder=3)

    # 真值轨迹（黑）
    ax.plot(f[:, 0], f[:, 1],
            color=GT_COLOR, lw=3.5, zorder=3)

    # 预测轨迹（jet 渐变，蓝→红，时间越长颜色越暖）
    pts  = p.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs,
                          cmap=plt.get_cmap('jet'),
                          norm=plt.Normalize(0, len(segs)),
                          lw=4.0, alpha=0.9, zorder=4)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)

    # 主车 3D 车模
    ego_yaw = yaw_from_traj(h)
    draw_car(ax, h[-1, 0], h[-1, 1], ego_yaw[-1],
             color=EGO_COLOR, scale=1.1, zorder=10)


def render_attn(ax, hist_np, others_np, attn_ego):
    """
    注意力图：高斯热力图标注每辆邻居车的注意力强度
    attn_ego [N] ego 对每个节点（含自身）的注意力权重
    """
    h  = to_plot(hist_np)
    cx, cy = h[-1, 0], h[-1, 1]

    setup_axis(ax, cx, cy)

    # Neighbors: heatmap + car marker at last frame only
    max_w = attn_ego[1:].max() if len(attn_ego) > 1 else 1e-6
    for ni, ov in enumerate(others_np):
        if len(ov) < 1:
            continue
        ov_p = to_plot(ov)
        last = ov_p[-1]
        if not (cx - 35 <= float(last[0]) <= cx + 35):
            continue
        w = attn_ego[ni + 1] if (ni + 1) < len(attn_ego) else 0.0
        draw_heatmap(ax, float(last[0]), float(last[1]), w, max_w)
        if len(ov_p) >= 2:
            yaw = math.atan2(float(ov_p[-1,1] - ov_p[-2,1]),
                             float(ov_p[-1,0] - ov_p[-2,0]))
        else:
            yaw = 0.0
        draw_car(ax, float(last[0]), float(last[1]), yaw,
                 color=OTH_COLOR, scale=1.1, zorder=3)

    # 主车
    ego_yaw = yaw_from_traj(h)
    draw_car(ax, h[-1, 0], h[-1, 1], ego_yaw[-1],
             color=EGO_COLOR, scale=1.1, zorder=10)


# ==========================================
# 8. 出图函数
# ==========================================
def save_single(hist_np, fut_np, pred_np, others_np, attn_ego,
                ade_val, fde_val, sample_idx, save_dir):
    """
    每个样本输出两张独立 PNG：
      *_pred.png  —— 预测轨迹图
      *_attn.png  —— 注意力热力图
    """
    stem = f"sample_{sample_idx:05d}_ade{ade_val:.3f}"

    # ── 预测图 ──
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    render_pred(ax, hist_np, fut_np, pred_np, others_np)

    # jet 色条
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap('jet'), norm=plt.Normalize(vmin=1, vmax=FUTURE_LEN / 10)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02,
                        ticks=np.linspace(1, FUTURE_LEN / 10, 5))
    cbar.ax.set_ylabel('Prediction time (s)',
                       fontsize=12, fontfamily='Times New Roman')
    cbar.ax.yaxis.set_tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, stem + '_pred.svg'),
                format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── 注意力图 ──
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    render_attn(ax, hist_np, others_np, attn_ego)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, stem + '_attn.svg'),
                format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_figure4_grid(samples, save_path):
    """
    论文 Figure 4 风格：2×2 拼图（预测图）+ 图例 + jet 色条
    samples: list of dict，每个含 hist/fut/pred/others 字段
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(bottom=0.22, hspace=0.35, wspace=0.15)

    for ax, s in zip(axes.flat, samples):
        ax.set_facecolor(BG_COLOR)
        render_pred(ax, s['hist'], s['fut'], s['pred'], s['others'])

    # 图例
    legend_elems = [
        Line2D([0], [0], color=HIST_COLOR, lw=3.5,
               label='History Trajectory'),
        Line2D([0], [0], color=GT_COLOR,   lw=3.5,
               label='Ground Truth Prediction'),
    ]
    fig.legend(handles=legend_elems,
               loc='lower center', bbox_to_anchor=(0.35, 0.10),
               ncol=2, frameon=True,
               facecolor=BG_COLOR, edgecolor='#D8D0C0',
               prop={'family': 'Times New Roman', 'size': 18})

    # jet 色条
    cbar_ax = fig.add_axes([0.65, 0.12, 0.22, 0.015])
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap('jet'),
        norm=plt.Normalize(vmin=1, vmax=FUTURE_LEN / 10)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                        ticks=np.linspace(1, FUTURE_LEN / 10, 5))
    cbar.ax.set_title('Prediction Time (s)',
                      fontsize=16, fontfamily='Times New Roman', pad=10)
    cbar.ax.set_xticklabels(
        [f'{v:.1f}' for v in np.linspace(1, FUTURE_LEN / 10, 5)],
        fontfamily='Times New Roman', fontsize=14
    )

    plt.figtext(0.5, 0.02,
                'Figure 4. ISTA-Net Model Trajectory Prediction',
                ha='center', fontsize=22, fontfamily='Times New Roman')
    fig.savefig(save_path, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def save_figure3_grid(samples, save_path):
    """论文 Figure 3 风格：2×2 注意力拼图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), dpi=200)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    for ax, s in zip(axes.flat, samples):
        ax.set_facecolor(BG_COLOR)
        render_attn(ax, s['hist'], s['others'], s['attn'])

    plt.figtext(0.5, 0.02,
                'Figure 3. Visualization of Spatial Attention Weights',
                ha='center', fontsize=22, fontfamily='Times New Roman')
    fig.savefig(save_path, format='svg', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ==========================================
# 9. 归一化统计 + 推理封装
# ==========================================
def run_inference(model, dataset, idx, feat_mean, feat_std, delta_std):
    history, history_graph, future = dataset[idx]

    pred_np, ade_val, fde_val, attn_ego = predict_best(
        model, history, history_graph,
        feat_mean, feat_std, delta_std, future
    )

    hist_np    = history.numpy()[:, 0, :2]      # [T, 2]  ego 历史
    fut_np     = future.numpy()                  # [50, 2]

    others_np = []
    for n in range(1, history.shape[1]):
        mask = history[:, n, 9].numpy() > 0.5
        if mask.sum() < 2:
            continue
        # 只取末尾连续有效帧，避免中间断开后再出现零点导致乱飞线段
        indices_valid = np.where(mask)[0]
        # 找末尾最长连续段：从最后一个有效帧往前找连续的
        last = indices_valid[-1]
        start = last
        for k in range(len(indices_valid) - 2, -1, -1):
            if indices_valid[k] == indices_valid[k + 1] - 1:
                start = indices_valid[k]
            else:
                break
        contiguous = history[start:last + 1, n, :2].numpy()
        if len(contiguous) >= 2:
            others_np.append(contiguous)

    return {
        'hist':   hist_np,
        'fut':    fut_np,
        'pred':   pred_np,
        'others': others_np,
        'attn':   attn_ego,
        'ade':    float(ade_val),
        'fde':    float(fde_val),
        'idx':    idx,
    }


# ==========================================
# 10. 主流程
# ==========================================
def main():
    print("加载模型...")
    model = TrajectoryGNNTransformer(
        in_channels=IN_CHANNELS, embed_dim=EMBED_DIM,
        lstm_hidden=LSTM_HIDDEN, num_heads=NUM_HEADS,
        ff_dim=FF_DIM, num_layers=NUM_LAYERS,
        future_len=FUTURE_LEN, num_modes=NUM_MODES,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载完成。")

    feat_mean, feat_std, delta_std = compute_statistics(TRAIN_DIR)
    feat_mean = feat_mean.to(DEVICE)
    feat_std  = feat_std.to(DEVICE)
    delta_std = delta_std.to(DEVICE)

    test_ds = NGSIMDataset(TEST_DIR)
    print(f"测试集：{len(test_ds)} 个样本")

    # ── 扫描，按 minADE 排序 ──
    scan_n  = min(len(test_ds), 600)
    indices = np.random.permutation(len(test_ds))[:scan_n].tolist()

    print(f"扫描 {scan_n} 个样本...")
    scores = []
    for idx in indices:
        s = run_inference(model, test_ds, idx,
                          feat_mean, feat_std, delta_std)
        scores.append(s)

    scores.sort(key=lambda x: x['ade'])

    # ── 全量独立 PNG（前 NUM_PLOTS 个最优）──
    best_scores = scores[:NUM_PLOTS]
    print(f"\n输出 {len(best_scores)} 张独立图到 {SAVE_DIR}/...")
    for rank, s in enumerate(best_scores):
        save_single(
            s['hist'], s['fut'], s['pred'], s['others'], s['attn'],
            s['ade'], s['fde'], s['idx'], SAVE_DIR
        )
        if (rank + 1) % 10 == 0:
            print(f"  {rank+1}/{len(best_scores)} 完成")

    # ── 论文拼图（取 ADE 最优的 4 个）──
    fig4_samples = scores[:4]
    save_figure4_grid(
        fig4_samples,
        os.path.join(SAVE_DIR, "Figure_4_Prediction.svg")
    )
    save_figure3_grid(
        fig4_samples,
        os.path.join(SAVE_DIR, "Figure_3_Attention.svg")
    )
    print("论文拼图已保存。")

    # ── 汇总指标 ──
    all_ade = [s['ade'] for s in scores]
    all_fde = [s['fde'] for s in scores]
    print(f"\n扫描集合汇总（N={scan_n}）：")
    print(f"  mean minADE = {np.mean(all_ade):.4f} m")
    print(f"  mean minFDE = {np.mean(all_fde):.4f} m")
    print(f"  best minADE = {np.min(all_ade):.4f} m")
    print(f"\n全部图片保存至 {SAVE_DIR}/")


if __name__ == "__main__":
    main()