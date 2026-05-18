import random
import envs.package as pkg
import numpy as np
import os, time

out_of_bound_reward = -0.2
best_rewards = 0.0
flag_2 = True

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0  # 记录调用 runlume 的次数

def runlume_6(
    stru_name='pra_6.txt',
    data_name="farfile_reflection_6.txt",
    model_file="jones_model_origin_6.fsp",
    mother_script="cal_farfield_data.lsf",
):
    def add_quota(string):
        return '\"' + string + '\"'
    
    file_root = "C:/data"
    stru_name = os.path.join(file_root, stru_name)
    data_name = os.path.join(file_root, data_name)
    model = os.path.join(file_root, model_file)
    fdtd_solutions = 'C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
    fsp_file = model
    lsf_file = "C:/data/script_6.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    return True

CD_global_best = 0
rl_lr_global_best = 0
m_global_best = np.inf
time_intervent = 0

def find_two_positions_by_argmin(sequence, threshold=0.1, min_distance=20):
    """
    使用argmin找到最小值位置，然后向左右搜索满足条件的最小值位置
    """
    
    # 使用argmin找到最小值位置
    min_pos = np.argmin(sequence)
    min_value = sequence[min_pos]
    
    # 检查最小值是否足够小
    if min_value >= threshold:
        return None, None
    
    # 初始化结果
    best_left_pos = None
    best_right_pos = None
    best_left_value = float('inf')
    best_right_value = float('inf')
    
    # 向左搜索
    left_start = min_pos - min_distance
    if left_start >= 0:
        for i in range(left_start, -1, -1):
            if sequence[i] < threshold and sequence[i] < best_left_value:
                best_left_pos = i
                best_left_value = sequence[i]
    
    # 向右搜索
    right_start = min_pos + min_distance
    if right_start < len(sequence):
        for i in range(right_start, len(sequence)):
            if sequence[i] < threshold and sequence[i] < best_right_value:
                best_right_pos = i
                best_right_value = sequence[i]
    
    # 返回结果：min_pos作为第一个位置，选择左右两侧找到的最小值作为第二个位置
    if best_left_pos is not None and best_right_pos is not None:
        # 两侧都找到，选择值更小的
        if best_left_value <= best_right_value:
            return best_left_pos, min_pos
        else:
            return min_pos, best_right_pos
    elif best_left_pos is not None:
        return best_left_pos, min_pos
    elif best_right_pos is not None:
        return min_pos, best_right_pos
    else:
        return None, None

def reward_6(data):

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
    m = (dis_real + dis_imag) / 2

    pos1, pos2 = find_two_positions_by_argmin(m, threshold=0.08, min_distance=20)

    # 如果找不到合适的两个点 - 给最低奖励
    if pos1 is None or pos2 is None:
        return 1 / (m[40] + m[120]) + 9 # 固定的很低的奖励
    print(f"real pos:{pos1},{pos2}")
    # 计算 CD 强度（正确的相对值计算）
    r_lr_1 = r_lr_real[pos1]**2 + r_lr_imag[pos1]**2
    r_rl_1 = r_rl_real[pos1]**2 + r_rl_imag[pos1]**2
    r_rr_1 = r_rr_real[pos1]**2 + r_rr_imag[pos1]**2
    
    r_lr_2 = r_lr_real[pos2]**2 + r_lr_imag[pos2]**2
    r_rl_2 = r_rl_real[pos2]**2 + r_rl_imag[pos2]**2
    r_rr_2 = r_rr_real[pos2]**2 + r_rr_imag[pos2]**2

    # 正确的CD计算公式
    cd1 = abs(r_lr_1 - r_rl_1) / (r_lr_1 + r_rl_1 + 2 * r_rr_1)
    cd2 = abs(r_lr_2 - r_rl_2) / (r_lr_2 + r_rl_2 + 2 * r_rr_2)
    print(f"real cd:",cd1, cd2)
    # CD强度奖励（相对值，通常在0-1之间）
    cd_reward = cd1 * cd2

    # 总奖励：只要找到两个零点就给高基础奖励，再加上质量加成
    base_reward = 9.0  # 找到两个零点的基础奖励
    quality_bonus =  cd_reward * 1000
    
    total_reward = base_reward + quality_bonus

    # 更新最佳结果
    if total_reward > best_rewards:
        best_rewards = total_reward
        with open("C:\\data\\best_result_6.txt", 'w') as f:
            f.write(f"Best Parameters: {data['pra']}\n")
            f.write(f"Best Reward: {best_rewards:.6f}\n")
            f.write(f"CD1: {cd1:.6f}, CD2: {cd2:.6f}\n")
            f.write(f"M1: {m[pos1]:.6f}, M2: {m[pos2]:.6f}\n")
            f.write(f"Pos1: {pos1}, Pos2: {pos2}\n")

    return total_reward

def run_onetime_6(pra):
    global simulation_count
    filepra = ','.join(str(p) for p in pra)
   
    filepath = os.path.join("C:\\data\\pra", filepra) + '.txt'
    pra = pra[:10]
    if os.path.exists(filepath):
        data = np.loadtxt(filepath)
    else:
        np.savetxt('C:\\data\\pra_6.txt', pra)
        runlume_6()
        if not os.path.exists("C:\\data\\farfile_reflection_6.txt"):
            simulation_count += 1 
            return out_of_bound_reward
        data = np.loadtxt("C:\\data\\farfile_reflection_6.txt")
        os.remove("C:\\data\\farfile_reflection_6.txt")  # 删除文件

    simulation_count += 1 
    pattern = pkg.PatternRects(pra)
    stru = pattern.get_details(op=data)
    if stru is None:
        return out_of_bound_reward
    
    return reward_6(stru)

# -------------------------------------------- #
# judge_flag用于实现判别器
judge_flag = False
judge_flag = True
if judge_flag:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # from judge.CD_inference import CustomResUNet
    # judger_model = CustomResUNet().to(device)
    # judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_0719.pth"))
    # judger_model.eval()

    # from judge.FCN import FCNRegressor
    # judger_model = FCNRegressor().to(device)
    # judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_fcn_0729.pth"))
    # judger_model.eval()

    # from judge.whole_trans import EdgeLengthTransformerRegressor
    # judger_model = EdgeLengthTransformerRegressor().to(device)
    # judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_trans_full_sequence_0904.pth"))
    # judger_model.eval()

    from judge.multi_trans import EdgeLengthTransformerRegressor
    judger_model = EdgeLengthTransformerRegressor().to(device)
    judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_trans_dual_m_0824.pth"))
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
        
        return torch.from_numpy(normalized).float().to(device)

def get_pattern_6(pra):  # ✅ 修正为 get_pattern_6
    pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10 = pra
    pattern = np.zeros((400, 400), dtype=int)
    for xi in range(pattern.shape[0]):
        for yj in range(pattern.shape[1]):
            if xi >= pra1 and xi <= pra1 + pra3:
                if yj >= pra10 + pra9 + pra6 + pra5 and \
                        yj <= pra10 + pra9 + pra6 + pra5 + pra2:
                    pattern[xi, yj] = 1
            if xi >= pra1 and xi <= pra1 + pra4:
                if yj >= pra10 + pra9 + pra6 and \
                        yj <= pra10 + pra9 + pra6 + pra5:
                    pattern[xi, yj] = 1
            if xi >= pra7 and xi <= pra7 + pra8:
                if yj >= pra10 and yj <= pra10 + pra9:
                    pattern[xi, yj] = 1
    return pattern

def run_onetime_total_new_6(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global flag_2
    global simulation_count
    pra_short = pra[:10]  

    # 生成 pattern
    pattern = get_pattern_6(pra_short)  # ✅ 修正函数名
    # 转换为模型输入格式 (B, C, W, W)
    input_tensor = torch.from_numpy(pattern).float().unsqueeze(0).unsqueeze(0).to(device)

    # 使用 judger 模型进行预判
    with torch.no_grad():
        predicted_reward = judger_model(input_tensor).item()
        
    # 设置阈值（根据模型输出范围设定）
    threshold = 0.40
    if predicted_reward < threshold:
        return predicted_reward + 1  
    print("judger:", predicted_reward)

    #否则继续执行 FDTD 模拟
    np.savetxt('C:\\data\\pra_6.txt', pra_short)
    runlume_6()
    if not os.path.exists("C:\\data\\farfile_reflection_6.txt"): 
        simulation_count += 1 
        return out_of_bound_reward
    
    simulation_count += 1   

    data = np.loadtxt("C:\\data\\farfile_reflection_6.txt")
    os.remove("C:\\data\\farfile_reflection_6.txt")  # 删除文件

    flag_2 = True

    pattern_obj = pkg.PatternRects(pra_short)
    stru = pattern_obj.get_details(op=data)

    if stru is None:
        return out_of_bound_reward

    return reward_6(stru)

def run_onetime_trans(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global flag_2
    global simulation_count
    pra_short = pra[:10]  

    input_tensor = torch.tensor(pra_short, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        
        tensor_value = judger_model(normalize(input_tensor)).cpu()
    # print(tensor_value.shape)

    # pos1, pos2 = find_two_positions_by_argmin(tensor_value.squeeze(0), threshold=0.1, min_distance=20)
    # if pos1 == None: 
    #     return 0.1
    # print(f"judger:{pos1},{pos2}")
    cd1, cd2 = tensor_value.squeeze(0)
    
    if cd1 < 0.3 and cd2 < 0.3:
        return cd1 * cd2 * 100
    print(cd1, cd2)
    

    np.savetxt('C:\\data\\pra_6.txt', pra_short)
    runlume_6()
    if not os.path.exists("C:\\data\\farfile_reflection_6.txt"): 
        simulation_count += 1 
        return out_of_bound_reward
    
    simulation_count += 1   

    data = np.loadtxt("C:\\data\\farfile_reflection_6.txt")
    os.remove("C:\\data\\farfile_reflection_6.txt")

    flag_2 = True

    pattern_obj = pkg.PatternRects(pra_short)
    stru = pattern_obj.get_details(op=data)

    if stru is None:
        return out_of_bound_reward

    return reward_6(stru)

def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_6.txt")
    pra_int = pras.astype(int)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        if not judge_flag:
            reward_all.append(run_onetime_6(pra_int))
        else:
            reward_all.append(run_onetime_trans(pra_int))  # 或者 run_onetime_total_new_6

    with open("C://data//reward_history_6.txt", 'a') as file:
        for w in reward_all:
            file.write(f"{pra_int}\n{w}\n")
    np.savetxt("C:\\data\\reward_6.txt", np.array(reward_all))