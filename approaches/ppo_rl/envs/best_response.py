# best_response.py

import os
import numpy as np
import envs.package as pkg

# 全局变量（仅在本模块内使用）
_best_rewards = -np.inf
_best_records = []
_simulation_count = 0
_out_of_bound_reward = -1  # 无效结构的默认 reward

def boundary_test(pra):
    c = (pra[0] >= 30 and pra[0] <= 160 and
         pra[2] >= 40 and pra[2] <= 80 and
         pra[3] >= 80 and pra[3] <= 340 and
         pra[0] + pra[3] <= 370 and
         pra[6] >= 30 and pra[6] <= 150 and
         pra[7] >= 40 and pra[7] <= 340 and
         pra[6] + pra[7] <= 340 and
         pra[9] >= 30 and pra[9] <= 100 and
         pra[8] >= 40 and pra[8] <= 100 and
         pra[5] >= 40 and pra[5] <= 100 and
         pra[4] >= 40 and pra[4] <= 100 and
         pra[1] >= 40 and pra[1] <= 100 and
         pra[9] + pra[8] + pra[5] + pra[4] + pra[1] <= 370)
    return c

def runlume_6(
    stru_name='pra_6.txt',
    data_name="farfile_reflection_6.txt",
    model_file="jones_model_origin_6.fsp",
    mother_script="cal_farfield_data.lsf"
):
    def add_quota(string):
        return '"' + string + '"'

    file_root = "C:/data"
    stru_name = os.path.join(file_root, stru_name)
    data_name = os.path.join(file_root, data_name)
    model = os.path.join(file_root, model_file)
    fdtd_solutions = 'C:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
    fsp_file = model
    lsf_file = "C:/data/script_6.lsf"
    cmd = " ".join([add_quota(fdtd_solutions), fsp_file, "-nw -run", lsf_file])
    os.system(cmd)
    return True

def _run_data(pra_short):
    global _simulation_count
    np.savetxt('C:\\data\\pra_6.txt', pra_short, fmt='%d')
    runlume_6()

    _simulation_count += 1
    if not os.path.exists("C:\\data\\farfile_reflection_6.txt"):
        return _out_of_bound_reward

    data = np.loadtxt("C:\\data\\farfile_reflection_6.txt")
    os.remove("C:\\data\\farfile_reflection_6.txt")

    pattern = pkg.PatternRects(pra_short)
    stru = pattern.get_details(op=data)

    if stru is None:
        return _out_of_bound_reward

    return _reward_6(stru, pra_short)

def _reward_6(data, pra):
    global _best_rewards, _best_records, _simulation_count
    wave_l = 88
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
        wave_l = 44

    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
    r = dis_real[wave_l]
    i = dis_imag[wave_l]
    m = (r + i) / 2
    c = CD[wave_l]
    reward = c

    if reward > _best_rewards:
        _best_rewards = reward
        with open("C:\\data\\best_result_br.txt", 'w') as f:
            f.write(f"m:{m}\n")
            f.write(f"CD:{c}\n")
            f.write(f"lr:{r_lr[wave_l]}\n")
            f.write(f"rl:{r_rl[wave_l]}\n")
            f.write(f"rr:{r_rr_ll[wave_l]}\n")
            f.write(f"wavelength:{wave_length[wave_l]}nm\n")
            f.write(f"Best Parameters: {list(pra)}\n")
            f.write(f"Best Reward: {_best_rewards:.6f}\n")

        _best_records.append((_simulation_count, _best_rewards))
        if _best_records:
            np.savetxt("C:\\data\\records_6.txt", np.array(_best_records), fmt='%d %.6f')

    return reward

def _single_parameter_search(pra_base, search_index, search_range, description="Single Parameter Search"):
    global _best_rewards, _best_records, _simulation_count

    print(f"[{description}] Searching parameter index {search_index}")
    print(f"Search range: {search_range}")

    best_local_pra = pra_base.copy()
    best_local_reward = -np.inf

    for val in search_range:
        current_pra = pra_base.copy()
        current_pra[search_index] = val

        if not boundary_test(current_pra):
            reward = _out_of_bound_reward
            print(f"[{description}] Invalid structure: {current_pra}, reward: {reward}")
        else:
            reward = _run_data(np.array(current_pra))

        print(f"[{description}] Parameters: {current_pra}, Reward: {reward}, count: {_simulation_count}")

        with open("C:/data/all_rewards_6.txt", "a") as f:
            f.write(f"{current_pra} -> {reward}\n")

        if reward > best_local_reward:
            best_local_reward = reward
            best_local_pra = current_pra.copy()

    return best_local_pra, best_local_reward

def run_best_response_search(initial_pra, search_indices):
    """
    执行两阶段 Best Response 参数搜索
    
    Args:
        initial_pra (list): 初始参数列表，长度为10
        search_indices (list): 要搜索的参数索引，如 [0,1,6,7]
    
    Returns:
        dict: 包含最终参数、最优 reward、总仿真次数等信息
    """
    global _best_rewards, _best_records, _simulation_count
    # 重置全局状态（每次搜索独立）
    _best_rewards = -np.inf
    _best_records = []
    _simulation_count = 0

    os.makedirs("C:/data", exist_ok=True)
    open("C:/data/all_rewards_6.txt", "w").close()  # 清空历史

    current_pra = list(initial_pra)

    print("=== 第一次粗搜索 ===")
    for idx in search_indices:
        print(f"\n--- 搜索参数 index {idx} ---")
        base_val = current_pra[idx]
        if idx == 0:
            search_range = list(range(30, 161, 4))
        elif idx == 1:
            search_range = list(range(40, 101, 4))
        elif idx == 6:
            search_range = list(range(30, 151, 4))
        elif idx == 7:
            search_range = list(range(40, 341, 4))
        else:
            search_range = list(range(max(30, base_val - 20), min(200, base_val + 21), 4))

        best_pra, best_reward = _single_parameter_search(
            current_pra, idx, search_range, f"Coarse Search - Parameter {idx}"
        )
        current_pra = best_pra.copy()
        print(f"参数 {idx} 搜索完成，当前最优参数: {current_pra}")

    pra_after_coarse = current_pra.copy()
    print(f"\n第一次粗搜索后最优参数: {pra_after_coarse}")

    print("\n=== 第二次精细搜索 ===")
    current_pra = pra_after_coarse.copy()

    for idx in search_indices:
        print(f"\n--- 精细搜索参数 index {idx} ---")
        best_val = current_pra[idx]
        bounds = {0: (30, 160), 1: (40, 100), 6: (30, 150), 7: (40, 340)}
        min_val, max_val = bounds.get(idx, (30, 200))
        start = max(min_val, best_val - 12)
        end = min(max_val, best_val + 12)
        fine_search_range = list(range(start, end + 1, 4))

        best_pra, best_reward = _single_parameter_search(
            current_pra, idx, fine_search_range, f"Fine Search - Parameter {idx}"
        )
        current_pra = best_pra.copy()
        print(f"参数 {idx} 精细搜索完成，当前最优参数: {current_pra}")

    final_pra = current_pra.copy()
    print(f"\n=== Best Response 搜索完成 ===")
    print(f"最终最优参数: {final_pra}")
    print(f"总共仿真次数: {_simulation_count}")
    print(f"最终 best reward: {_best_rewards}")

    return {
        "final_pra": final_pra,
        "best_reward": _best_rewards,
        "simulation_count": _simulation_count,
        "best_records": _best_records.copy()
    }