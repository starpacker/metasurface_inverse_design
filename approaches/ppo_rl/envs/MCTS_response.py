import os
import numpy as np
import envs.package as pkg
import random
from math import log, sqrt

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
        with open("C:/data/all_rewards_6.txt", "a") as f:
            f.write(f"{pra_short.tolist()} -> {_out_of_bound_reward}\n")
        return _out_of_bound_reward

    data = np.loadtxt("C:\\data\\farfile_reflection_6.txt")
    os.remove("C:\\data\\farfile_reflection_6.txt")

    pattern = pkg.PatternRects(pra_short)
    stru = pattern.get_details(op=data)

    if stru is None:
        reward = _out_of_bound_reward
        with open("C:/data/all_rewards_6.txt", "a") as f:
            f.write(f"{pra_short.tolist()} -> {reward}\n")
        return reward

    reward = _reward_6(stru, pra_short)
    with open("C:/data/all_rewards_6.txt", "a") as f:
        f.write(f"{pra_short.tolist()} -> {reward}\n")
    return reward

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

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # tuple of parameter values for search_indices up to depth
        self.parent = parent
        self.action = action
        self.children = {}  # action -> child node
        self.total_reward = 0.0
        self.visits = 0
        self.depth = len(state)

def is_terminal(node, K):
    return node.depth == K

def get_full_pra(state, base_pra, search_indices):
    pra = list(base_pra)
    for i, idx in enumerate(search_indices):
        pra[idx] = state[i]
    return pra

def evaluate(node, base_pra, search_indices):
    full_pra = get_full_pra(node.state, base_pra, search_indices)
    if not boundary_test(full_pra):
        return _out_of_bound_reward
    return _run_data(np.array(full_pra))

def tree_policy(node, possible_actions_list, base_pra, search_indices):
    K = len(search_indices)
    if is_terminal(node, K):
        # Evaluate only once per terminal node
        if node.visits == 0:
            reward = evaluate(node, base_pra, search_indices)
            node.total_reward = reward
            node.visits = 1
        else:
            reward = node.total_reward
        return reward

    # Select best action using UCT
    depth = node.depth
    actions = possible_actions_list[depth]
    uct_list = []
    for action in actions:
        if action in node.children:
            child = node.children[action]
            n = max(node.visits, 1)
            ni = max(child.visits, 1)
            q = child.total_reward / child.visits if child.visits > 0 else 0
            explore = 1.4 * sqrt(log(n) / ni)  # C = 1.4
            uct = q + explore
        else:
            uct = float('inf')  # Prioritize unexplored
        uct_list.append((uct, action))

    # Sort descending UCT, pick first
    uct_list.sort(key=lambda x: x[0], reverse=True)
    selected_action = uct_list[0][1]

    # Create child if not exists
    if selected_action not in node.children:
        new_state = node.state + (selected_action,)
        child = MCTSNode(new_state, parent=node, action=selected_action)
        node.children[selected_action] = child

    child = node.children[selected_action]

    # Recurse
    reward = tree_policy(child, possible_actions_list, base_pra, search_indices)

    # Backpropagate
    node.total_reward += reward
    node.visits += 1

    return reward

def get_best_state(node, possible_actions_list, K):
    state = list(node.state)
    current = node
    for d in range(node.depth, K):
        if not current.children:
            # Fallback to first possible action
            action = possible_actions_list[d][0]
        else:
            best_action = max(current.children.keys(), key=lambda a: current.children[a].total_reward / max(current.children[a].visits, 1))
            action = best_action
        state.append(action)
        if action in current.children:
            current = current.children[action]
        else:
            break
    return tuple(state)

def get_coarse_actions(idx):
    if idx == 0:
        return list(range(30, 161, 8))
    elif idx == 1:
        return list(range(40, 101, 8))
    elif idx == 6:
        return list(range(30, 151, 8))
    elif idx == 7:
        return list(range(40, 341, 8))
    return []

def get_fine_actions(idx, center_val):
    bounds = {0: (30, 160), 1: (40, 100), 6: (30, 150), 7: (40, 340)}
    min_val, max_val = bounds.get(idx, (30, 200))
    start = max(min_val, center_val - 12)
    end = min(max_val, center_val + 12)
    # Snap start to multiple of 4
    start = start - (start % 4) if start % 4 != 0 else start
    return list(range(start, end + 1, 4))

def run_mcts_response_search(initial_pra, search_indices, coarse_budget=200, fine_budget=300):
    """
    执行两阶段 MCTS 参数搜索
    
    Args:
        initial_pra (list): 初始参数列表，长度为10
        search_indices (list): 要搜索的参数索引，如 [0,1,6,7]
        coarse_budget (int): 粗搜索预算（仿真次数）
        fine_budget (int): 精细搜索预算（仿真次数）
    
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

    K = len(search_indices)

    print("=== MCTS 第一次粗搜索 ===")
    root_state = tuple(initial_pra[idx] for idx in search_indices)
    root = MCTSNode(root_state)
    possible_actions_list = [get_coarse_actions(idx) for idx in search_indices]

    for _ in range(coarse_budget):
        tree_policy(root, possible_actions_list, initial_pra, search_indices)

    best_state_coarse = get_best_state(root, possible_actions_list, K)
    pra_after_coarse = list(initial_pra)
    for i, val in enumerate(best_state_coarse):
        pra_after_coarse[search_indices[i]] = val
    print(f"粗搜索后最优参数: {pra_after_coarse}")

    print("\n=== MCTS 第二次精细搜索 ===")
    # Reset globals for fine search tracking, but simulations continue counting
    root_state_fine = tuple(pra_after_coarse[idx] for idx in search_indices)
    root_fine = MCTSNode(root_state_fine)
    fine_possible_actions_list = []
    for i, idx in enumerate(search_indices):
        center = pra_after_coarse[idx]
        actions = get_fine_actions(idx, center)
        fine_possible_actions_list.append(actions)

    for _ in range(fine_budget):
        tree_policy(root_fine, fine_possible_actions_list, pra_after_coarse, search_indices)

    best_state_fine = get_best_state(root_fine, fine_possible_actions_list, K)
    final_pra = list(pra_after_coarse)
    for i, val in enumerate(best_state_fine):
        final_pra[search_indices[i]] = val

    print(f"\n=== MCTS 搜索完成 ===")
    print(f"最终最优参数: {final_pra}")
    print(f"总共仿真次数: {_simulation_count}")
    print(f"最终 best reward: {_best_rewards}")

    return {
        "final_pra": final_pra,
        "best_reward": _best_rewards,
        "simulation_count": _simulation_count,
        "best_records": _best_records.copy()
    }