import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from model import TrajectoryGNNTransformer
from train import TrajectoryDataset

# ==========================================
# 0. 配置与日志设置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])

# 全局美学配置：高级羊皮纸背景色
BG_COLOR = '#F5F2EB'
GRID_COLOR = '#E2DCD0'
MAIN_COLOR = '#D32F2F' # 主车红色
OTHER_COLOR = '#909090' # 周围车辆灰色

# ==========================================
# 1. 基础绘图与坐标工具
# ==========================================
def transform_coordinates(data):
    """
    恢复水平向右行驶！
    水平轴放 Y(纵向行驶里程)，垂直轴放 -X(横向车道偏移)
    """
    if isinstance(data, list): data = np.array(data)
    if len(data) == 0: return data
    return np.stack([data[:, 1], -data[:, 0]], axis=1)

def get_yaw_from_trajectory(traj):
    if len(traj) < 2: return np.zeros(1)
    dx, dy = traj[1:, 0] - traj[:-1, 0], traj[1:, 1] - traj[:-1, 1]
    dx, dy = np.concatenate([dx, [dx[-1]]]), np.concatenate([dy, [dy[-1]]])
    return np.arctan2(dy, dx)

def draw_heatmap_with_probability(ax, center_x, center_y, weight, max_weight):
    """极致平滑的高斯热力图，水平方向拉长贴合车身"""
    if max_weight == 0: return
    rel_weight = weight / max_weight 
    if rel_weight <= 0.05: return 
    
    x = np.linspace(center_x - 18, center_x + 18, 120)
    y = np.linspace(center_y - 10, center_y + 10, 120)
    X, Y = np.meshgrid(x, y)
    
    # 水平方向扩散更大 (spread_x > spread_y)
    spread_x = 8.0 + 4.0 * rel_weight
    spread_y = 3.0 + 2.0 * rel_weight
    Z = np.exp(-(((X - center_x)**2) / spread_x + ((Y - center_y)**2) / spread_y))
    Z = Z * rel_weight
    
    # 从完全透明到深红的平滑渐变
    colors = [(0.75, 0.10, 0.10, 0.0), (0.75, 0.10, 0.10, 0.95)]
    custom_reds = mcolors.LinearSegmentedColormap.from_list('TransparentReds', colors)
    
    contour = ax.contourf(X, Y, Z, levels=60, cmap=custom_reds, vmin=0, vmax=1.0, zorder=2)
    
    # 兼容新老版本 Matplotlib 的 SVG 抗锯齿修复
    if hasattr(contour, 'collections'):
        for c in contour.collections:
            c.set_edgecolor("face")
    else:
        contour.set_edgecolor("face")

def draw_detailed_car(ax, x, y, yaw, color='red', scale=1.1, alpha=1.0, zorder=10):
    """带有阴影和车窗的高级 3D 俯视模型"""
    LENGTH = 4.8 * scale
    WIDTH = 2.0 * scale
    tr = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData

    # 底部阴影
    shadow_offset = -0.3
    ax.add_patch(FancyBboxPatch((-LENGTH/2 + shadow_offset, -WIDTH/2 + shadow_offset), 
                                LENGTH, WIDTH, boxstyle="round,pad=0,rounding_size=0.4", 
                                ec='none', fc='black', alpha=0.25 * alpha, transform=tr, zorder=zorder-2))

    # 轮胎
    wheel_len = LENGTH * 0.18
    wheel_wid = WIDTH * 0.22
    for wx, wy in [(LENGTH/2 - wheel_len, WIDTH/2 - wheel_wid/2), (LENGTH/2 - wheel_len, -WIDTH/2 - wheel_wid/2),
                   (-LENGTH/2, WIDTH/2 - wheel_wid/2), (-LENGTH/2, -WIDTH/2 - wheel_wid/2)]:
        ax.add_patch(Rectangle((wx, wy), wheel_len, wheel_wid, color='#222222', transform=tr, zorder=zorder-1))

    # 车身
    ax.add_patch(FancyBboxPatch((-LENGTH/2, -WIDTH/2), LENGTH, WIDTH, 
                                boxstyle="round,pad=0,rounding_size=0.4",
                                ec='#333333', lw=1.2, fc=color, alpha=alpha, transform=tr, zorder=zorder))

    # 车窗玻璃
    ax.add_patch(FancyBboxPatch((LENGTH*0.08, -WIDTH*0.35), LENGTH*0.22, WIDTH*0.7, 
                                boxstyle="round,pad=0,rounding_size=0.08", 
                                ec='black', lw=0.8, fc='#2A2A2A', alpha=alpha*0.9, transform=tr, zorder=zorder+1))
    ax.add_patch(FancyBboxPatch((-LENGTH*0.35, -WIDTH*0.35), LENGTH*0.15, WIDTH*0.7, 
                                boxstyle="round,pad=0,rounding_size=0.05", 
                                ec='black', lw=0.8, fc='#2A2A2A', alpha=alpha*0.9, transform=tr, zorder=zorder+1))

# ==========================================
# 2. 核心统一渲染器 
# ==========================================
def render_single_axis(ax, data, plot_type='pred'):
    hist_rot = transform_coordinates(data['hist'])
    fut_rot = transform_coordinates(data.get('fut', []))
    pred_rot = transform_coordinates(data.get('pred', []))
    
    center_x, center_y = hist_rot[-1, 0], hist_rot[-1, 1]
    
    # 恢复水平长视野 (-35m 到 +35m)
    x_lim_min, x_lim_max = center_x - 35, center_x + 35
    y_lim_min, y_lim_max = center_y - 8, center_y + 8
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_aspect('equal')
    
    # 羊皮纸背景网格
    ax.grid(True, linestyle='-', linewidth=0.6, color=GRID_COLOR, zorder=0)
    
    # 水平车道线 (虚线)
    base_lane_y = np.round(center_y / 3.75) * 3.75
    for offset in [-7.5, -3.75, 0, 3.75, 7.5]:
        ax.axhline(base_lane_y + offset, color='#666666', linestyle='--', linewidth=1.5, dashes=(5, 5), zorder=1)
        
    valid_attn = data.get('attn_weights', [])
    max_weight = max(valid_attn) if valid_attn else 1e-6
        
    for i, ov in enumerate(data['others']):
        ov_rot = transform_coordinates(ov)
        
        # 彻底清洗 Padding 产生的 (0,0) 原点坐标！消除乱飞的直线！
        valid_mask = (ov_rot[:, 0] != 0.0) | (ov_rot[:, 1] != 0.0)
        ov_rot_clean = ov_rot[valid_mask]
        
        if len(ov_rot_clean) < 2: continue 
        curr_x = ov_rot_clean[-1, 0]
        if curr_x < x_lim_min or curr_x > x_lim_max: continue
            
        yaw = get_yaw_from_trajectory(ov_rot_clean)
        
        # 纯净的注意力图
        if plot_type == 'attn' and 'attn_weights' in data:
            draw_heatmap_with_probability(ax, ov_rot_clean[-1,0], ov_rot_clean[-1,1], data['attn_weights'][i], max_weight)
            
        # 纯净的周围车辆灰线 (只在预测图画)
        if plot_type == 'pred':
            ax.plot(ov_rot_clean[:,0], ov_rot_clean[:,1], color='#999999', linewidth=2.0, alpha=0.5, zorder=2)
            
        draw_detailed_car(ax, ov_rot_clean[-1,0], ov_rot_clean[-1,1], yaw[-1], color=OTHER_COLOR, scale=1.1, zorder=3)
        
    ego_yaw = get_yaw_from_trajectory(hist_rot)
    
    if plot_type == 'pred':
        ax.plot(hist_rot[:,0], hist_rot[:,1], color='#777777', linewidth=3.5, zorder=3)
        if len(fut_rot) > 0:
            ax.plot(fut_rot[:,0], fut_rot[:,1], color='#222222', linewidth=3.5, zorder=3)
        if len(pred_rot) > 0:
            points = np.array([pred_rot[:,0], pred_rot[:,1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, len(segments)), linewidth=4.0, alpha=0.9, zorder=4)
            lc.set_array(np.arange(len(segments)))
            ax.add_collection(lc)
            
    draw_detailed_car(ax, hist_rot[-1,0], hist_rot[-1,1], ego_yaw[-1], color=MAIN_COLOR, scale=1.1, zorder=10)
    
    # 完美的物理对应坐标标签：水平是 y/m，垂直是 x/m
    ax.set_xlabel('y/m', fontsize=16, fontfamily='Times New Roman', labelpad=10)
    ax.set_ylabel('x/m', fontsize=16, fontfamily='Times New Roman', labelpad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontname('Times New Roman')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.2)

# ==========================================
# 3. 工作流：输出独立 SVG 模式
# ==========================================
def generate_single_sample_svgs(data, sample_idx, ade_val, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    fig_attn, ax_attn = plt.subplots(figsize=(10, 5), dpi=300)
    fig_attn.patch.set_facecolor(BG_COLOR)  
    ax_attn.set_facecolor(BG_COLOR)         
    
    render_single_axis(ax_attn, data, plot_type='attn')
    plt.title(f"Spatial Attention Weights - Sample: {sample_idx}", fontsize=16, fontfamily='Times New Roman', pad=15)
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx:04d}_ade_{ade_val:.3f}_attn.svg"), format='svg', bbox_inches='tight', facecolor=fig_attn.get_facecolor())
    plt.close(fig_attn)
    
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5), dpi=300)
    fig_pred.patch.set_facecolor(BG_COLOR)
    ax_pred.set_facecolor(BG_COLOR)
    
    render_single_axis(ax_pred, data, plot_type='pred')
    plt.title(f"Trajectory Prediction - Sample: {sample_idx} | ADE: {ade_val:.3f}m", fontsize=16, fontfamily='Times New Roman', pad=15)
    plt.savefig(os.path.join(save_dir, f"sample_{sample_idx:04d}_ade_{ade_val:.3f}_pred.svg"), format='svg', bbox_inches='tight', facecolor=fig_pred.get_facecolor())
    plt.close(fig_pred)

def generate_figure_3_svg(samples_data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)
    for ax, data in zip(axes.flat, samples_data):
        ax.set_facecolor(BG_COLOR)
        render_single_axis(ax, data, plot_type='attn')
    plt.figtext(0.5, 0.02, "Figure 3. Visualization of Spatial Attention Weights", ha="center", fontsize=22, fontfamily='Times New Roman')
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def generate_figure_4_svg(samples_data, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(bottom=0.25, hspace=0.35, wspace=0.15)
    for ax, data in zip(axes.flat, samples_data):
        ax.set_facecolor(BG_COLOR)
        render_single_axis(ax, data, plot_type='pred')
        
    legend_elements = [
        Line2D([0], [0], color='#777777', lw=3.5, label='History Trajectory'),
        Line2D([0], [0], color='#222222', lw=3.5, label='Ground Truth Prediction')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.35, 0.12), ncol=2, 
               frameon=True, facecolor=BG_COLOR, edgecolor='#D8D0C0', prop={'family':'Times New Roman','size':18})
    
    cbar_ax = fig.add_axes([0.65, 0.14, 0.22, 0.015])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_title('Prediction Time (s)', fontsize=16, fontfamily='Times New Roman', pad=10)
    cbar.ax.set_xticklabels(['1', '2', '3', '4', '5'], fontfamily='Times New Roman', fontsize=14)
    
    plt.figtext(0.5, 0.02, "Figure 4. ISTA-Net Model Trajectory Prediction", ha="center", fontsize=22, fontfamily='Times New Roman')
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# ==========================================
# 4. 数据调度与主逻辑
# ==========================================
def ade(pred, gt): return torch.mean(torch.norm(pred - gt, dim=-1)).item()

def get_sample_data(model, dataset, idx, device):
    history, adj, last_pos, future = dataset[idx]
    
    history_t = history.unsqueeze(0).to(device)
    adj_t = adj.unsqueeze(0).to(device)
    last_pos_t = last_pos.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred, attn_weights = model(history_t, adj_t, last_pos_t, return_attn=True)
    
    hist_np = history_t[0, :, 0, :2].detach().cpu().numpy()
    fut_np = future.detach().cpu().numpy()
    pred_np = pred[0].detach().cpu().numpy()
    ego_attn_to_others = attn_weights[0, 0, 1:].detach().cpu().numpy()
    
    others_np = []
    valid_attn = []
    num_nodes = history_t.shape[2]
    for n in range(1, num_nodes):
        track = history_t[0, :, n, :2].cpu().numpy()
        if np.sum(np.abs(track)) > 0.1: 
            others_np.append(track)
            valid_attn.append(ego_attn_to_others[n-1])
            
    return {'hist': hist_np, 'fut': fut_np, 'pred': pred_np, 'others': others_np, 'attn_weights': valid_attn}

def main(args):
    device = torch.device("cpu")
    save_root = os.path.expanduser(args.output_dir)
    os.makedirs(save_root, exist_ok=True)
    
    model = TrajectoryGNNTransformer().to(device)
    model.load_state_dict(torch.load(os.path.expanduser(args.model_path), map_location=device))
    
    data_root = os.path.expanduser(args.data_dir)
    test_dataset = TrajectoryDataset(os.path.join(data_root, "test"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    GENERATE_ALL_SAMPLES = True  
    CHOSEN_IDS = [12, 45, 88, 102] 
    
    model.eval()
    all_metrics = []
    logging.info(">>> 正在扫描测试集计算指标...")
    with torch.no_grad():
        for batch_idx, (history, adj, last_pos, future) in enumerate(test_loader):
            history, adj, last_pos, future = history.to(device), adj.to(device), last_pos.to(device), future.to(device)
            pred = model(history, adj, last_pos)
            for i in range(pred.shape[0]):
                s_pred, s_future = pred[i:i+1], future[i:i+1]
                idx = batch_idx * test_loader.batch_size + i
                all_metrics.append((ade(s_pred, s_future), idx))
    
    all_metrics.sort(key=lambda x: x[0]) 
    
    if GENERATE_ALL_SAMPLES:
        total_samples = len(all_metrics)
        preview_dir = os.path.join(save_root, "All_Test_Samples_Separate_SVG")
        logging.info(f">>> [全量生成模式] 准备生成所有 {total_samples} 个样本的独立 SVG 图...")
        for i in range(total_samples):
            ade_val, idx = all_metrics[i]
            sample_data = get_sample_data(model, test_dataset, idx, device)
            generate_single_sample_svgs(sample_data, idx, ade_val, preview_dir)
            if (i+1) % 50 == 0:
                logging.info(f"已生成 {i+1}/{total_samples} 组 SVG ...")
        logging.info("✅ 极其精美的复古工程图风格 SVG 图像全部生成完毕！")
    
    else:
        logging.info(f">>> [出图模式] 正在提取你指定的 ID: {CHOSEN_IDS} 并渲染 2x2 拼图...")
        samples_data = []
        for idx in CHOSEN_IDS:
            samples_data.append(get_sample_data(model, test_dataset, idx, device))
            
        logging.info(">>> 正在生成 Figure 3. SVG...")
        generate_figure_3_svg(samples_data, os.path.join(save_root, "Figure_3_Attention_Final.svg"))
        logging.info(">>> 正在生成 Figure 4. SVG...")
        generate_figure_4_svg(samples_data, os.path.join(save_root, "Figure_4_Prediction_Final.svg"))
        logging.info("✅ 论文终图生成完毕，赶快插入论文看看效果吧！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="~/Desktop/processed_trajectory/best_model.pth")
    parser.add_argument('--data_dir', type=str, default="~/Desktop/processed_trajectory")
    parser.add_argument('--output_dir', type=str, default="~/Desktop/eval_results")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()
    main(args)