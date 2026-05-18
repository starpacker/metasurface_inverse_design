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

def runlume_3(
    stru_name='pra_3.txt',
    data_name="farfile_reflection_3.txt",
    model_file="jones_model_origin_3.fsp",
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
    lsf_file = "C:/data/script_3.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
    os.system(cmd)
    # print(cmd)
    return True

CD_global_best = 0
rl_lr_global_best = 0
m_global_best = np.inf
time_intervent = 0

def reward_3(data):
    # wave_l = 88
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


    r_lr = r_lr_real ** 2 + r_lr_imag ** 2
    r_rl = r_rl_real ** 2 + r_rl_imag ** 2
    r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2

    if wave_length[wave_l] > 650:
        # wave_l = 44
        wave_l = 42

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
    reward = 0
    # if m > 0.5 or c<0.2:
    #     reward = c
    #     # return c
    # elif m > 0.02:
    #     # return
    #     reward =  0.5*(0.5 - m + 0.6*c)+(np.angle(r_lr_angle)[100]+3.14)/10
    # else:
    #     # return 
    #     reward = 1.0+c

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
        best_rewards  = reward
        with open("C:\\data\\best_result_3.txt", 'w') as f:
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
        np.savetxt(f"C:\\data\\records_3.txt", np.array(best_records), fmt='%d %.6f')

    # write CD history
    file = open("C://data//CD_history_3.txt", 'a')
    file.writelines(str(c)+'\n')
    file.close()

    return reward

def run_onetime_3(pra):
    print("start simulating")
    global simulation_count
    filepra = ','.join(str(p) for p in pra)
   
    filepath = os.path.join("C:\\data\\pra", filepra) + '.txt'
    pra = pra[:10]
    if os.path.exists(filepath):
        data = np.loadtxt(filepath)
    else:
        np.savetxt('C:\\data\\pra_3.txt', pra)
        runlume_3()
        if not os.path.exists("C:\\data\\farfile_reflection_3.txt"):
            print("simulate error")
            simulation_count += 1 
            return out_of_bound_reward
        data = np.loadtxt("C:\\data\\farfile_reflection_3.txt")
        os.remove("C:\\data\\farfile_reflection_3.txt")  # 删除文件

    simulation_count += 1 
    pattern = pkg.PatternRects(pra)
    stru = pattern.get_details(op=data)
    if stru is None:
        return out_of_bound_reward
    
    return reward_3(stru)

# -------------------------------------------- #
# judge_flag用于实现判别器
judge_flag = False
# judge_flag = True

def get_pattern_3(pra):  # ✅ 修正为 get_pattern_3
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


def get_reward():
    reward_all = []
    pras = np.loadtxt("C:\\data\\pra_all_3.txt")
    pra_int = pras.astype(int)
    # print(pras)
    if pra_int[0] == 0:
        reward_all.append(out_of_bound_reward)
    else:
        reward_all.append(run_onetime_3(pra_int))
    with open("C://data//reward_history_3.txt", 'a') as file:
        for w in reward_all:
            file.write(f"{pra_int}\n{w}\n")
    np.savetxt("C:\\data\\reward_3.txt", np.array(reward_all))