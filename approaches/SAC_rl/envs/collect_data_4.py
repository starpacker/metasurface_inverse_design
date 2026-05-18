import random
import envs.package as pkg
import numpy as np
import os, time
from collections import deque

# Judger 训练缓冲区 - 使用 deque 实现最大长度限制
MAX_BUFFER_SIZE = 128
train_interval = 150
batchsize = 32
# MAX_BUFFER_SIZE = 1
# train_interval = 1
# batchsize = 1
judger_training_buffer = deque(maxlen=MAX_BUFFER_SIZE)

wave_l = 88
out_of_bound_reward = -0.2
best_rewards = 0.0

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0  # 记录调用 runlume 的次数

# 🔧 动态阈值相关参数
initial_threshold = 0.0    # 初始阈值
final_threshold = 0.7      # 最终阈值
threshold_increase_rate = 0.2  # 每 threshold_update_interval 次仿真增加的量
threshold_update_interval = MAX_BUFFER_SIZE   # 每多少次仿真更新一次阈值
current_threshold = initial_threshold  # 当前阈值

def update_threshold():
    """动态更新阈值"""
    global current_threshold, simulation_count
    # 计算应该增加多少次
    increases = simulation_count // threshold_update_interval
    new_threshold = initial_threshold + increases * threshold_increase_rate
    current_threshold = min(new_threshold, final_threshold)  # 不超过最终阈值
    return current_threshold


def runlume_4(
        stru_name='pra_4.txt',
        data_name="farfile_reflection_4.txt",
        model_file="jones_model_origin_4.fsp",
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
    lsf_file = "C:/data/script_4.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    return True

CD_global_best = 0
rl_lr_global_best = 0
m_global_best = np.inf
time_intervent = 0

def reward_4(data):
    wave_l = 88
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

    r_lr = r_lr_real ** 2 + r_lr_imag ** 2
    r_rl = r_rl_real ** 2 + r_rl_imag ** 2
    r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2
    r_lr_angle = r_lr_real + 1j * r_lr_imag

    if wave_length[wave_l] > 650:
        wave_l = 44

    rl_lr = np.abs(np.log(r_lr[wave_l]) - np.log(r_rl[wave_l]))
    if r_rl[wave_l] > r_lr[wave_l]:
        rl_rr = np.log(r_rl[wave_l]) - np.log(r_rr_ll[wave_l])
    else:
        rl_rr = np.log(r_lr[wave_l]) - np.log(r_rr_ll[wave_l])
    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
    r = dis_real[wave_l]
    i = dis_imag[wave_l]
    m = (r + i) / 2
    c = CD[wave_l]

    if m > 0.5 or c < 0.2:
        reward = c
    elif m > 0.02:
        reward = 0.5 * (0.5 - m + 0.6 * c) + (np.angle(r_lr_angle)[100] + 3.14) / 10
    else:
        reward = 1.0 + c

    if rl_lr > 0 and rl_rr > 0:
        reward += rl_lr
    if rl_lr > 1 and rl_rr > 1:
        reward += rl_lr * 1
    if rl_lr > 2 and rl_rr > 1:
        reward += rl_lr * 1
    if rl_lr > 2 and rl_rr > 2:
        reward += rl_rr * 1
    if rl_lr > 3 and rl_rr > 2:
        reward += rl_rr * 1
    if rl_lr > 3 and rl_rr > 3:
        reward += rl_lr * 1
    if rl_lr > 4 and rl_rr > 3:
        reward += rl_lr * 2
    if rl_lr > 4 and rl_rr > 4:
        reward += rl_rr * 5

    if reward > best_rewards:
        best_rewards = reward
        with open("C:\\data\\best_result_4.txt", 'w') as f:
            f.write(f"m:{m}\n")
            f.write(f"CD:{c}\n")
            f.write(f"lr:{r_lr[wave_l]}\n")
            f.write(f"rl:{r_rl[wave_l]}\n")
            f.write(f"rr:{r_rr_ll[wave_l]}\n")
            f.write(f"rl_lr:{rl_lr}\n")
            f.write(f"rl_rr:{rl_rr}\n")
            f.write(f"wavelength:{wave_length[wave_l]}nm\n")
            f.write(f"Best Parameters: {data['pra']}\n")
            f.write(f"Best Reward: {best_rewards:.6f}\n")

        global best_records, simulation_count
        best_records.append((simulation_count, best_rewards))
        np.savetxt(f"C:\\data\\records_True_4.txt", np.array(best_records), fmt='%d %.6f')

    file = open("C://data//CD_history_4.txt", 'a')
    file.writelines(str(c) + '\n')
    file.close()

    return c,reward

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

    # 简单训练几步
    optimizer = torch.optim.Adam(judger_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

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

def save_judger_model(path="C:/light_mappo-main-2/judge/best_model_trans_online_4.pth"):
    """保存当前训练好的 Judger 模型"""
    torch.save(judger_model.state_dict(), path)

def run_onetime_total_new_4(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global simulation_count
    pra_short = pra[:10]

    # 生成 pattern
    pattern = np.array(pra[:10], dtype=np.float32)
    input_tensor = torch.from_numpy(pattern).float().unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_reward = judger_model(torch.from_numpy(normalize(input_tensor)).float().to(device)).item()


    # 更新并获取当前阈值
    current_threshold = update_threshold()
    print(f"📊 当前仿真次数: {simulation_count}, 当前阈值: {current_threshold:.3f}")

    if predicted_reward < current_threshold:
        return predicted_reward + 1
    print("judger:", predicted_reward)

    # judger_training_buffer.append((pra_short, 0.1))
    # # 检查是否需要训练 Judger
    # train_judger_online()
    # return 0.1


    np.savetxt('C:\\data\\pra_4.txt', pra_short)
    runlume_4()
    if not os.path.exists("C:\\data\\farfile_reflection_4.txt"):
        simulation_count += 1
        return out_of_bound_reward

    simulation_count += 1

    data = np.loadtxt("C:\\data\\farfile_reflection_4.txt")
    
    # 生成逗号分隔的文件名
    filepath = "C:\\data\\pra\\" + ",".join(map(str, pra_short)) + ".txt"
    if not os.path.exists(filepath):
        np.savetxt(filepath, data)

    os.remove("C:\\data\\farfile_reflection_4.txt")  # 删除文件


    pattern = pkg.PatternRects(pra_short)
    stru = pattern.get_details(op=data)

    if stru is None:
        return out_of_bound_reward
    real_cd, real_reward = reward_4(stru)

    judger_training_buffer.append((pra_short, real_cd))

    # 检查是否需要训练 Judger
    train_judger_online()

    return real_reward

def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_4.txt")
    pra_int = pras.astype(int)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        reward_all.append(run_onetime_total_new_4(pra_int))

    file = open("C://data//reward_history_4.txt", 'a')

    for w in reward_all:
        file.writelines(str(pra_int) + '\n')
        file.writelines(str(w) + '\n')
    file.close()
    np.savetxt("C:\\data\\reward_4.txt", np.array(reward_all))