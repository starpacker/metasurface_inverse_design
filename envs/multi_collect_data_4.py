import random
import envs.package as pkg
import numpy as np
import os, time
from scipy.signal import find_peaks
from collections import deque


# Judger 训练缓冲区 - 使用 deque 实现最大长度限制
MAX_BUFFER_SIZE = 128
train_interval = 150
batchsize = 32
judger_training_buffer = deque(maxlen=MAX_BUFFER_SIZE)

out_of_bound_reward = -0.2
best_rewards = 0.0
threshold_cd_extra = 0.2

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0  # 记录调用 runlume 的次数

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
        0: (30, 160),
        1: (40, 100),
        2: (40, 80),
        3: (80, 340),
        4: (40, 100),
        5: (40, 100),
        6: (30, 150),
        7: (40, 340),
        8: (40, 100),
        9: (30, 100)
    }

optimizer = torch.optim.Adam(judger_model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

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
    global judger_training_buffer, judger_model, simulation_count

    # 每 train_interval 次仿真训练一次，且缓冲区大小 >= batchsize
    if simulation_count % train_interval != 0 or len(judger_training_buffer) < batchsize:
        return

    print(f"🔥 开始微调 Judger 模型... 当前缓冲区大小: {len(judger_training_buffer)}")

    # 提取参数和奖励
    pra_list = []
    rewards = []
    for pra_short, reward in judger_training_buffer:
           

        pra_list.append(normalize(pra_short))
        rewards.append(reward)
    # print(pra_list)
    # 转换为 tensor
    all_inputs = torch.from_numpy(np.array(pra_list)).float().to(device)  # shape: [N, 10]
    all_targets = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(device)  # shape: [N, 1]

    # 创建 Dataset 和 DataLoader
    dataset = torch.utils.data.TensorDataset(all_inputs, all_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)


    judger_model.train()
    total_loss = 0
    num_epochs = 3

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            # print(batch_inputs.shape)
            outputs = judger_model(batch_inputs.squeeze(1))
            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

    avg_loss = total_loss / num_epochs
    judger_model.eval()

    print(f"✅ Judger 微调完成，平均 loss = {avg_loss:.6f}")

    save_judger_model()
    judger_training_buffer.clear()  # 清空 buffer

def save_judger_model(path="C:/light_mappo-main-2/judge/best_model_trans_online_8.pth"):
    """保存当前训练好的 Judger 模型"""
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

    judger_training_buffer.append((pra_short, real_m))

    # 检查是否需要训练 Judger
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