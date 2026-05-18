import random
import envs.package as pkg
import numpy as np
import os, time

# wave_l = 88
out_of_bound_reward = -0.2
best_rewards = 0.0
flag_2 = True

# 新增记录 best reward 和对应 simulation 次数的列表
best_records = []
simulation_count = 0  # 记录调用 runlume 的次数

def runlume_5(
    stru_name='pra_5.txt',
    data_name="farfile_reflection_5.txt",
    model_file="jones_model_origin_5.fsp",
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
    lsf_file = "C:/data/script_5.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    return True

CD_global_best = 0
rl_lr_global_best = 0
m_global_best = np.inf
time_intervent = 0

def reward_5(data):
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
    r_lr_angle = r_lr_real + 1j*r_lr_imag

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
    m = (r + i)/2
    c = CD[wave_l]
    
    if m > 0.5 or c<0.2:
        reward = c
        # return c
    elif m > 0.02:
        # return
        reward =  0.5*(0.5 - m + 0.6*c)+(np.angle(r_lr_angle)[100]+3.14)/10
    else:
        # return 
        reward = 1.0+c

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
    if judge_flag:
        print("judge real CD:", c )

    if reward > best_rewards:
        best_rewards  = reward
        with open("C:\\data\\best_result_5.txt", 'w') as f:
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
        # 记录当前仿真次数和对应的 best reward
        best_records.append((simulation_count, best_rewards))
        # 保存 best reward 到文件
        np.savetxt(f"C:\\data\\records_{judge_flag}_5.txt", np.array(best_records), fmt='%d %.6f')

    # write CD history
    file = open("C://data//CD_history_5.txt", 'a')
    file.writelines(str(c)+'\n')
    file.close()

    return reward

def run_onetime_5(pra):
    global simulation_count
    pra = pra[:10]
    np.savetxt('C:\\data\\pra_5.txt', pra)
    runlume_5()
    if not os.path.exists("C:\\data\\farfile_reflection_5.txt"):
        simulation_count += 1 
        return out_of_bound_reward
    data = np.loadtxt("C:\\data\\farfile_reflection_5.txt")
    os.remove("C:\\data\\farfile_reflection_5.txt")  # 删除文件

    simulation_count += 1 
    pattern = pkg.PatternRects(pra)
    stru = pattern.get_details(op=data)
    if stru is None:
        return out_of_bound_reward
    
    return reward_5(stru)

# -------------------------------------------- #
# judge_flag用于实现判别器
judge_flag = False
# judge_flag = True
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

    from judge.transformer import EdgeLengthTransformerRegressor
    judger_model = EdgeLengthTransformerRegressor().to(device)
    judger_model.load_state_dict(torch.load("C:/light_mappo-main-2/judge/best_model_trans_0821.pth"))
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

def run_onetime_trans(pra):  # 这个函数选择不依靠任何老数据，纯依赖judger和FDTD进行PPO算法
    global flag_2
    global simulation_count
    pra_short = pra[:10]  

    input_tensor = torch.tensor(pra_short, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_reward = judger_model(normalize(input_tensor)).item()
        
    threshold = 0.70
    if predicted_reward < threshold:
        return predicted_reward + 1  
    print("judger:", predicted_reward)

    np.savetxt('C:\\data\\pra_5.txt', pra_short)
    runlume_5()
    if not os.path.exists("C:\\data\\farfile_reflection_5.txt"): 
        simulation_count += 1 
        return out_of_bound_reward
    
    simulation_count += 1   

    data = np.loadtxt("C:\\data\\farfile_reflection_5.txt")
    os.remove("C:\\data\\farfile_reflection_5.txt")

    flag_2 = True

    pattern_obj = pkg.PatternRects(pra_short)
    stru = pattern_obj.get_details(op=data)

    if stru is None:
        return out_of_bound_reward

    return reward_5(stru)

def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_5.txt")
    pra_int = pras.astype(int)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        if not judge_flag:
            reward_all.append(run_onetime_5(pra_int))
        else:
            reward_all.append(run_onetime_trans(pra_int))  # 或者 run_onetime_total_new_5

    with open("C://data//reward_history_5.txt", 'a') as file:
        for w in reward_all:
            file.write(f"{pra_int}\n{w}\n")
    np.savetxt("C:\\data\\reward_5.txt", np.array(reward_all))