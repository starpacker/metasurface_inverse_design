import random
import envs.package as pkg
import numpy as np
import os, time
from scipy.signal import find_peaks
import torch

# ================= 配置区域 =================
# Judger 训练缓冲区配置
MAX_BUFFER_SIZE = 2000  # [修改] 增大 Buffer 容量，防止遗忘
train_interval = 150    # 每多少次仿真训练一次
batchsize = 32
out_of_bound_reward = -0.2
best_rewards = 0.0
threshold_cd_extra = 0.2
max_error_clip = 1.0    # [修改] Error Clipping 上限

# PER 参数
PER_ALPHA = 0.6         # 决定"多大程度上"侧重于高误差样本 (0=纯随机, 1=完全按误差)
PER_BETA = 0.4          # IS Weights 的初始值，用于修正偏差
PER_EPSILON = 1e-6      # 防止误差为 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0 

# ----------------- 简单优先经验回放 Buffer (无 SumTree 版) -----------------
class SimplePERBuffer:
    """
    适用于小数据量的优先经验回放缓冲区。
    不使用 SumTree，直接使用 numpy 数组计算概率，适合数据量 < 10000 的场景。
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = [] # 存储结构: [ [error, pra, target], ... ]
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, error, pra, target):
        """添加新样本，error 会被 clip"""
        # [修改] Error Clipping
        clipped_error = np.clip(abs(error), 0, max_error_clip)
        
        data = [clipped_error, pra, target]
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            # 简单的 FIFO 替换，也可以改为替换优先级最低的
            self.buffer[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """基于优先级采样，并返回 IS Weights"""
        N = len(self.buffer)
        
        # 1. 提取所有 error 并计算采样概率
        errors = np.array([item[0] for item in self.buffer])
        priorities = (errors + PER_EPSILON) ** self.alpha
        probs = priorities / priorities.sum()

        # 2. 根据概率采样索引
        indices = np.random.choice(N, batch_size, p=probs, replace=True) # 允许重复采样
        
        # 3. 提取 Batch 数据
        batch_data = [self.buffer[idx] for idx in indices]
        
        # 4. 计算 IS Weights (Focus on weakness 的修正)
        # w_j = (N * P_j) ^ -beta
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max() # 归一化权重，保持稳定性
        
        # 拆分数据返回
        b_pra = []
        b_target = []
        for item in batch_data:
            b_pra.append(item[1])
            b_target.append(item[2])
            
        return (indices, 
                np.array(b_pra), 
                np.array(b_target), 
                torch.tensor(weights, dtype=torch.float32).to(device))

    def update_priorities(self, indices, new_errors):
        """训练后更新被采样样本的优先级"""
        for idx, error in zip(indices, new_errors):
            clipped_error = np.clip(abs(error), 0, max_error_clip)
            self.buffer[idx][0] = clipped_error

# 初始化全局 Buffer
per_buffer = SimplePERBuffer(capacity=MAX_BUFFER_SIZE, alpha=PER_ALPHA, beta=PER_BETA)

# --------------------------------------------------------------------------
def runlume_8(
        stru_name='pra_8.txt',
        data_name="farfile_reflection_8.txt",
        model_file="jones_model_origin_8.fsp",
        mother_script="cal_farfield_data.lsf", ):
    # 结构，模型，lsf脚本
    def add_quota(string):
        return '\"' + string + '\"'

    file_root = "C:/data"
    stru_name = os.path.join(file_root, stru_name)
    data_name = os.path.join(file_root, data_name)
    model = os.path.join(file_root, model_file)
    fdtd_solutions = 'C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
    fsp_file = model
    lsf_file = "C:/data/script_8.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    return True

def find_valley_positions(sequence, threshold=0.1, min_distance=20):

    #    寻找 data 的最小值 < threshold
    #    等价于寻找 -data 的最大值 > -threshold
    inverted_data = -sequence
    height_limit = -threshold
    
    #    调用 find_peaks
    #    height: 峰的最低高度（对应原数据的最大允许值）
    #    distance: 峰与峰之间的最小水平距离
    peaks, _ = find_peaks(inverted_data, height=height_limit, distance=min_distance)
    
    # 4. 返回结果 (转换为 list)
    return peaks.tolist()

def reward_8(data):
    wave_l = 84
    global best_rewards
    wave_length = np.array(data['wavelength'])
    eig_state_1_real = np.array(data['eig_state_1_real'])
    eig_state_1_imag = np.array(data['eig_state_1_imag'])
    eig_state_2_real = np.array(data['eig_state_2_real'])
    eig_state_2_imag = np.array(data['eig_state_2_imag'])
    r_lr_real = np.array(data['r_lr_real'])
    r_lr_imag = np.array(data['r_lr_imag'])
    r_rl_real = np.array(data['r_rl_real'])
    r_rl_imag = np.array(data['r_rl_imag'])
    r_rr_real = np.array(data['r_rr_real'])
    r_rr_imag = np.array(data['r_rr_imag'])

    dis_real = np.abs(eig_state_1_real - eig_state_2_real)
    dis_imag = np.abs(eig_state_1_imag - eig_state_2_imag)

    if wave_length[wave_l] > 650:
        wave_l = 42

    m = (dis_real + dis_imag) / 2
    r_lr = r_lr_real ** 2 + r_lr_imag ** 2
    r_rl = r_rl_real ** 2 + r_rl_imag ** 2
    r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2
    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)

    peaks_indices = find_valley_positions(m, threshold=0.15, min_distance=20)

    if len(peaks_indices) < 2:
        reward = 1.0 / (m[40] + m[160] + 1e-6)
        return reward
    
    cd_values_at_peaks = CD[peaks_indices]
    sorted_local_indices = np.argsort(cd_values_at_peaks)[::-1]
    sorted_global_peaks = peaks_indices[sorted_local_indices]
    pos1 = sorted_global_peaks[0]
    pos2 = sorted_global_peaks[1]
    cd1 = CD[pos1]
    cd2 = CD[pos2]
    
    cd_reward = cd1 * cd2
    base_reward = 1.0
    quality_bonus = cd_reward * 1000
    
    total_reward = base_reward + quality_bonus

    if len(sorted_global_peaks) > 2:
        # 获取第3个及之后的全局位置
        remaining_positions = sorted_global_peaks[2:]
        
        for pos_n in remaining_positions:
            cd_n = CD[pos_n]
            if cd_n > threshold_cd_extra:
                total_reward += cd_n * 1000

    if total_reward > best_rewards:
        best_rewards = total_reward
        with open("C:\\data\\best_result_8.txt", 'w') as f:
            f.write(f"Best Parameters: {data['pra']}\n")
            f.write(f"Best Reward: {best_rewards:.6f}\n")
            f.write(f"CD1: {cd1:.6f}, CD2: {cd2:.6f}\n")
            f.write(f"M1: {m[pos1]:.6f}, M2: {m[pos2]:.6f}\n")
            f.write(f"Pos1: {pos1}, Pos2: {pos2}\n")


        global best_records, simulation_count
        best_records.append((simulation_count, best_rewards))
        np.savetxt(f"C:\\data\\records_True_8.txt", np.array(best_records), fmt='%d %.6f')

    return m, total_reward

# -------------------------------------------- #
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from judge.multi_trans import EdgeLengthTransformerRegressor
judger_model = EdgeLengthTransformerRegressor().to(device)
judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_trans_full_m_now.pth"))
judger_model.eval()
    # 参数边界定义
param_bounds = {
        0: (30, 160), 1: (40, 100), 2: (40, 80), 3: (80, 340), 4: (40, 100),
        5: (40, 100), 6: (30, 150), 7: (40, 340), 8: (40, 100), 9: (30, 100)
    }

# [修改] 全局优化器，使用较小的 LR 进行微调
optimizer = torch.optim.Adam(judger_model.parameters(), lr=1e-5)
# Loss 不需要 reduction='mean'，因为我们要手动乘 weights
criterion_none = torch.nn.MSELoss(reduction='none') 

def normalize(params):
        """
        将参数归一化到[0,1]范围
        params: [N, 10] 或 [10] (支持 numpy 数组或 torch Tensor，包括 CUDA Tensor)
        """
        # 检测输入类型并转换为 numpy 数组
        if isinstance(params, torch.Tensor):
            # 如果是 CUDA Tensor，先转移到 CPU 再转为 numpy
            if params.is_cuda:
                params = params.cpu()
            params = params.detach().numpy()
        else:
            params = np.array(params)
        original_shape = params.shape
        if len(original_shape) == 1:
            params = params.reshape(1, -1)
        
        normalized = np.zeros_like(params, dtype=np.float32)
        for i in range(len(param_bounds)):
            min_val, max_val = param_bounds[i]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
            normalized[:, i] = np.clip(normalized[:, i], 0, 1)
        
        return normalized

def train_judger_online():
    """
    基于 PER (Prioritized Experience Replay) 的训练循环
    Focus on weakness: 重点训练高误差样本
    """
    global judger_model, simulation_count, optimizer, per_buffer

    # 数据太少不训练
    if simulation_count % train_interval != 0 or len(per_buffer) < batchsize:
        return

    print(f"🔥 [Focus on Weakness] 微调 Judger... Buffer Size: {len(per_buffer)}")

    judger_model.train()
    
    # 每次触发训练时，进行多次小批量更新 (例如 5 次)，充分利用这次机会
    steps_per_train = 5
    total_loss = 0
    
    for _ in range(steps_per_train):
        # 1. 采样 (indices 用于后续更新 error, weights 用于修正 loss)
        indices, b_pra, b_target, is_weights = per_buffer.sample(batchsize)
        
        # 准备数据
        # b_pra 是 (Batch, 10), b_target 是 (Batch, Spectrum_Len)
        norm_inputs = []
        for p in b_pra:
            norm_inputs.append(normalize(p).squeeze())
            
        inputs_tensor = torch.from_numpy(np.array(norm_inputs)).float().to(device)
        targets_tensor = torch.from_numpy(b_target).float().to(device)
        
        # 2. 前向传播
        outputs = judger_model(inputs_tensor) # [Batch, Spectrum_Len]
        
        # 3. 计算 Loss (配合 IS Weights)
        # 先计算每个样本的 MSE: [Batch, Spectrum_Len] -> mean(dim=1) -> [Batch]
        loss_per_sample = torch.mean((outputs - targets_tensor) ** 2, dim=1)
        
        # 应用权重: Loss = mean( Weight * Error )
        weighted_loss = (loss_per_sample * is_weights).mean()
        
        # 4. 反向传播
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()
        
        # 5. [关键] 更新 Buffer 中的优先级 (基于新的预测误差)
        # detach() 很重要，转回 numpy
        new_errors = loss_per_sample.detach().cpu().numpy()
        per_buffer.update_priorities(indices, new_errors)

    judger_model.eval()
    print(f"✅ 微调完成. Avg Weighted Loss: {total_loss/steps_per_train:.6f}")
    
    # 注意：使用 PER 后，不需要 clear buffer，保留历史数据以防止遗忘

def save_judger_model(path="C:/light_mappo-main-2/judge/best_model_trans_online_8.pth"):
    torch.save(judger_model.state_dict(), path)

def run_onetime_total_new_8(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global simulation_count
    pra_short = pra[:10]

    input_tensor = torch.tensor(pra_short, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        
        tensor_value = judger_model(normalize(input_tensor)).cpu().squeeze(0)

    tensor_value_np = tensor_value.detach().numpy()

    peaks_indices = find_valley_positions(tensor_value_np, threshold=0.15, min_distance=20)

    judge_reject = False
    if len(peaks_indices) < 2:
        judge_reject = True

    if judge_reject:
        if random.random() < 0.1:  
            force_run = True
        else:
            # 真的跳过
            reward = 1.0 / (tensor_value_np[40] + tensor_value_np[160] + 1e-6)
            return reward

    np.savetxt('C:\\data\\pra_8.txt', pra_short)
    runlume_8()
    if not os.path.exists("C:\\data\\farfile_reflection_8.txt"):
        simulation_count += 1
        return out_of_bound_reward

    simulation_count += 1

    data = np.loadtxt("C:\\data\\farfile_reflection_8.txt")
    
    # 生成逗号分隔的文件名
    filepath = "C:\\data\\pra\\" + ",".join(map(str, pra_short)) + ".txt"
    if not os.path.exists(filepath):
        np.savetxt(filepath, data)

    os.remove("C:\\data\\farfile_reflection_8.txt")  # 删除文件


    pattern = pkg.PatternRects(pra_short)
    stru = pattern.get_details(op=data)

    if stru is None:
        return out_of_bound_reward
    real_m, real_reward = reward_8(stru)

    # --- [关键] 计算误差并存入 PER Buffer ---
    # 计算预测误差 (Surprise / Weakness 指标)
    # MSE 作为误差衡量
    current_prediction_error = np.mean((tensor_value_np - real_m) ** 2)
    # 存入 Buffer (Error 会被 clip)
    per_buffer.add(current_prediction_error, pra_short, real_m)

    # --- 触发训练 ---
    train_judger_online()


    return real_reward

def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_8.txt")
    pra_int = pras.astype(int)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        reward_all.append(run_onetime_total_new_8(pra_int))

    file = open("C://data//reward_history_8.txt", 'a')

    for w in reward_all:
        file.writelines(str(pra_int) + '\n')
        file.writelines(str(w) + '\n')
    file.close()
    np.savetxt("C:\\data\\reward_8.txt", np.array(reward_all))

