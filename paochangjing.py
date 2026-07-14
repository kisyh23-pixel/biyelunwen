import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter

# ====================== [模块 1] 全局参数配置 =======================
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
SAVE_DIR_MERGE = os.path.join(DESKTOP_PATH, "processed_trajectory_merging")
SAVE_DIR_FOLLOW = os.path.join(DESKTOP_PATH, "processed_trajectory_following")

for d in [SAVE_DIR_MERGE, SAVE_DIR_FOLLOW]:
    os.makedirs(os.path.join(d, "train"), exist_ok=True)
    os.makedirs(os.path.join(d, "test"), exist_ok=True)

# 学术标准时间窗
HIST_LEN = 30    
FUTURE_LEN = 50  
STEP = 5         

MAX_NEIGHBORS = 7  
MAX_NODES = MAX_NEIGHBORS + 1  
SPEED_THRESHOLD = 35.0  
MIN_SPEED = 0.5         

# 滤波与特征维度
WINDOW_LENGTH = 15  
POLY_ORDER = 3      
FEATURE_DIM = 10  # 终极维度: [X, Y, V, A, Dist_L, Dist_R, Lane_Type, Sin_H, Cos_H, Mask]

# ====================== [模块 2] 数据读取与单位转换 =======================
print("读取 NGSIM 数据集...")
df = pd.read_csv(r"C:/Users/MR/Desktop/NGSIM.csv")

df.rename(columns={
    'Vehicle ID': 'trackId', 'Frame ID': 'frameId', 
    'Local X': 'localX', 'Local Y': 'localY', 
    'Lane Identification': 'laneId', 'Vehicle Velocity': 'velocity'
}, inplace=True)

df['trackId'] = df['trackId'].astype(str).str.replace(' ', '', regex=False).astype(int)
df['frameId'] = df['frameId'].astype(str).str.replace(' ', '', regex=False).astype(int)

# 转换为纯净物理单位：米 (meters)
df[['localX', 'localY', 'velocity']] *= 0.3048

# ====================== [模块 3] 高阶物理特征计算与去噪 =======================
print("执行 S-G 滤波与 10 维特征工程计算...")

smoothed_dfs = []
for track_id, group in tqdm(df.groupby('trackId'), desc="Feature Engineering (Smoothing)"):
    if len(group) >= WINDOW_LENGTH:
        group = group.copy()
        
        # 1. 平滑坐标与速度
        for col in ['localX', 'localY', 'velocity']:
            group[col] = savgol_filter(group[col], WINDOW_LENGTH, POLY_ORDER)
            
        # 2. 运动学：加速度提取与平滑
        group['acceleration'] = np.gradient(group['velocity']) / 0.1
        group['acceleration'] = savgol_filter(group['acceleration'], WINDOW_LENGTH, POLY_ORDER)
        
        # 3. Heading (朝向) 提取与低速抗噪
        group['v_x'] = np.gradient(group['localX']) / 0.1
        group['v_y'] = np.gradient(group['localY']) / 0.1
        group['v_x'] = savgol_filter(group['v_x'], WINDOW_LENGTH, POLY_ORDER)
        group['v_y'] = savgol_filter(group['v_y'], WINDOW_LENGTH, POLY_ORDER)
        
        speed = np.sqrt(group['v_x']**2 + group['v_y']**2)
        yaw = np.arctan2(group['v_y'], group['v_x'])
        
        # 抹除极低速下的原地打转噪音，利用惯性前后填充
        yaw[speed < 0.5] = np.nan
        yaw = pd.Series(yaw).ffill().bfill().fillna(0.0).values 
        
        group['sin_h'] = np.sin(yaw)
        group['cos_h'] = np.cos(yaw)
        
        # 4. 场景语义
        group['lane_type'] = (group['laneId'] > 5).astype(float)
        
        smoothed_dfs.append(group)

# 组合平滑后的全局数据集
df = pd.concat(smoothed_dfs, ignore_index=True)

# 在全局平滑后，再提取车道中心线，并进行横向距离 Clip 防御！
print("基于平滑轨迹反推全局车道拓扑并计算偏移...")
lane_centers = df.groupby('laneId')['localX'].median().to_dict()

# 向量化计算全局横向偏移，完美防止负值空间扭曲
df['lane_center'] = df['laneId'].map(lane_centers).fillna(df['localX'])
df['dist_L'] = np.clip(df['localX'] - (df['lane_center'] - 1.85), 0.0, 3.7)
df['dist_R'] = np.clip((df['lane_center'] + 1.85) - df['localX'], 0.0, 3.7)

# ====================== [模块 4] 交通场景预筛选 =======================
print("分离 变道/汇入 (Merging) 与 纯跟车 (Following) 场景...")
lane_changes = df.groupby('trackId')['laneId'].nunique()

merging_vids = lane_changes[lane_changes > 1].index
following_vids = lane_changes[lane_changes == 1].index

df_merging = df[df['trackId'].isin(merging_vids)].copy()
df_following = df[df['trackId'].isin(following_vids)].copy()

# ====================== [模块 5] 核心张量切片与坐标系转换 =======================
def generate_samples(target_df, base_save_dir, is_merging_dataset=False):
    target_df.sort_values(by=["trackId", "frameId"], inplace=True)
    sample_id = 0
    train_dir = os.path.join(base_save_dir, "train")
    test_dir = os.path.join(base_save_dir, "test")

    print(f"\n正在构建全局帧级哈希字典 (O(1) Lookup Table)...")
    global_frame_dict = {fid: sub for fid, sub in df.groupby("frameId")}
    print("哈希字典构建完成！切片速度起飞")

    for track_id, group in tqdm(target_df.groupby("trackId"), desc="Generating Tensors"):
        
        # 【核心防御 1】：按 track_id 进行严格正交分割，彻底杜绝数据泄露！
        # 利用 track_id 末位进行 8:2 分割，同一辆车的所有滑窗必定进入同一个集合
        is_train_vehicle = (int(track_id) % 10) < 8
        folder = train_dir if is_train_vehicle else test_dir

        group = group.reset_index(drop=True)
        frame_ids = group["frameId"].values

        # 宏观长度校验与速度校验
        if len(frame_ids) < HIST_LEN + FUTURE_LEN: continue
        if np.any(group["velocity"].values > SPEED_THRESHOLD) or np.mean(group["velocity"].values) < MIN_SPEED: continue

        for start in range(0, len(group) - HIST_LEN - FUTURE_LEN + 1, STEP):
            sub = group.iloc[start: start + HIST_LEN + FUTURE_LEN]
            
            # 【核心防御 2】：局部窗口连续性严格校验，剔除丢帧残缺轨迹
            if not np.all(np.diff(sub["frameId"].values) == 1): 
                continue
                
            history, future = sub.iloc[:HIST_LEN], sub.iloc[HIST_LEN:]

            # 统一静态锚点
            anchor_x = history['localX'].iloc[-1]
            anchor_y = history['localY'].iloc[-1]

            # 【核心防御 3】：刚柔并济的”变道双重锁”，过滤蛇行废料
            if is_merging_dataset:
                current_lane = history['laneId'].iloc[-1]
                future_lat_shift = abs(future['localX'].iloc[-1] - anchor_x)
                lane_label_changed = future['laneId'].iloc[-1] != current_lane
                
                # 双重锁：横移突破2米(强物理变道) OR (横移突破1米 且 传感器确认跨线)
                is_real_lane_change = (future_lat_shift > 2.0) or ((future_lat_shift > 1.0) and lane_label_changed)
                
                if not is_real_lane_change: 
                    continue 

            frame_range = history["frameId"].values
            
            hist_tensor = np.zeros((HIST_LEN, MAX_NODES, FEATURE_DIM), dtype=np.float32)
            adj_tensor = np.zeros((HIST_LEN, MAX_NODES, MAX_NODES), dtype=np.float32)
            valid_sequence = True

            for t, f in enumerate(frame_range):
                f_df = global_frame_dict.get(f) 
                
                ego = group[group["frameId"] == f]
                if ego.empty or f_df is None or f_df.empty: 
                    valid_sequence = False; break
                ego = ego.iloc[0]

                # 写入主车特征 (静态锚点坐标系)
                hist_tensor[t, 0, :] = [
                    ego.localX - anchor_x, ego.localY - anchor_y, 
                    ego.velocity, ego.acceleration, 
                    ego.dist_L, ego.dist_R, ego.lane_type, 
                    ego.sin_h, ego.cos_h, 1.0 # Mask = 1.0
                ]

                # O(1) 寻找同帧邻居 
                others = f_df[f_df['trackId'] != track_id].copy()
                others["dist"] = np.sqrt((others["localX"] - ego.localX)**2 + (others["localY"] - ego.localY)**2)
                neighbors = others.nsmallest(MAX_NEIGHBORS, "dist")
                valid_nodes_count = 1 + len(neighbors)
                
                for i, (_, row) in enumerate(neighbors.iterrows()):
                    # 写入邻居车辆特征 (同处于全局静态锚点坐标系)
                    hist_tensor[t, i+1, :] = [
                        row.localX - anchor_x, row.localY - anchor_y,
                        row.velocity, row.acceleration,
                        row.dist_L, row.dist_R, row.lane_type, 
                        row.sin_h, row.cos_h, 1.0 # Mask = 1.0
                    ]
                
                # 向量化的 RBF 高斯核建图
                valid_xy = hist_tensor[t, :valid_nodes_count, :2]
                diff = valid_xy[:, np.newaxis, :] - valid_xy[np.newaxis, :, :]
                dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
                
                rbf_weights = np.exp(-dist_matrix / 10.0)
                rbf_weights[rbf_weights < 0.05] = 0.0
                adj_tensor[t, :valid_nodes_count, :valid_nodes_count] = rbf_weights

            if not valid_sequence: continue

            # 未来坐标的锚点对齐 (纯净的物理距离)
            future_xy = future[["localX", "localY"]].values.astype(np.float32)
            future_xy[:, 0] -= anchor_x
            future_xy[:, 1] -= anchor_y

            sample = {
                "trackId": int(track_id),
                "history": hist_tensor, "history_graph": adj_tensor, "future": future_xy             
            }
            np.save(os.path.join(folder, f"sample_{sample_id:05d}.npy"), sample)
            sample_id += 1
            
    return sample_id

print("\n--- 开始生成 变道/汇入 (Merging) 满血 10维 特征集 ---")
merged_count = generate_samples(df_merging, SAVE_DIR_MERGE, is_merging_dataset=True)
print(f"成功生成 {merged_count} 个顶级变道切片！")

print("\n--- 开始生成 纯跟车 (Following) 数据集 ---")
following_count = generate_samples(df_following, SAVE_DIR_FOLLOW, is_merging_dataset=False)
print(f"成功生成 {following_count} 个跟车切片！")

print("\n数据管线全部执行完毕！底层基建无懈可击，可以开启 Z-Score 训练了！")