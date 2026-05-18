import os
import numpy as np

import envs.package as pkg

import random
from math import log, sqrt
import copy
import math

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

import random
import math
import copy
import json
import os


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # List of parameters
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0  # Sum of rewards (for maximization)
        self.is_terminal = False

    def best_child(self, c=1.4):
        if not self.children:
            return None
        best_value = float('-inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                uct_value = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                uct_value = exploitation + exploration
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

    def select(self, c=1.4):
        current = self
        while current.children:
            current = current.best_child(c)
            if current is None:
                break
            if current.is_terminal:
                return current
        return current

    def expand(self):
        if self.is_terminal:
            return
        # Generate possible actions: modify one parameter by +1 or -1
        for i in range(len(self.state)):
            for delta in [-1, 1]:
                new_state = copy.deepcopy(self.state)
                new_state[i] += delta
                if boundary_test(new_state):  # Only expand valid states
                    child = MCTSNode(new_state, parent=self)
                    self.children.append(child)

    def simulate(self, simulation_depth=5):
        # Simple random rollout: perform random modifications up to depth, then evaluate
        sim_state = copy.deepcopy(self.state)
        for _ in range(simulation_depth):
            if random.random() < 0.5:  # 50% chance to stop early
                break
            i = random.randint(0, len(sim_state) - 1)
            delta = random.choice([-1, 1])
            sim_state[i] += delta
            if not boundary_test(sim_state):
                # Revert if invalid
                sim_state[i] -= delta
        # Evaluate the final state
        reward = _run_data(sim_state)
        return reward

    def backpropagate(self, reward):
        current = self
        while current:
            current.visits += 1
            current.value += reward  # For maximization
            current = current.parent

def mcts_search(root_state, iterations=1000, simulation_depth=5, c=1.4):
    root = MCTSNode(root_state)
    for _ in range(iterations):
        # Selection
        leaf = root.select(c)
        # Expansion
        if not leaf.is_terminal and leaf.visits == 0:
            leaf.expand()
            if leaf.children:
                leaf = random.choice(leaf.children)  # Rollout from a random child if expanded
        # Simulation
        if not leaf.is_terminal:
            reward = leaf.simulate(simulation_depth)
        else:
            reward = _run_data(leaf.state)  # If terminal, just evaluate
        # Backpropagation
        leaf.backpropagate(reward)
    
    # Find the best leaf (highest average reward)
    best_node = root
    current = root
    while current.children:
        best_child = max(current.children, key=lambda child: child.value / child.visits if child.visits > 0 else float('-inf'))
        if best_child.value / best_child.visits > best_node.value / best_node.visits:
            best_node = best_child
        current = best_child
    
    return best_node.state, best_node.value / best_node.visits

def load_best_params(filename="best_params.json"):
    """加载历史最佳参数和奖励，如果文件不存在则返回None"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data['best_reward'], data['best_pra']
    return None, None

def save_best_params(best_reward, best_pra, filename="C:\\data\\best_params.json"):
    """保存最佳参数和奖励到文件"""
    data = {
        'best_reward': best_reward,
        'best_pra': best_pra
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def run_mcts_with_update(init_pra, iterations=1000, simulation_depth=5, c=1.4, filename="best_params.json"):
    """
    运行MCTS搜索，从init_pra开始，尝试更新全局最佳参数。
    如果找到更好的reward，则更新并保存到文件。
    返回本次搜索的最佳参数、reward，以及当前全局最佳。
    """
    # 加载当前全局最佳
    current_best_reward, current_best_pra = load_best_params(filename)
    
    # 如果没有历史最佳，使用init_pra作为初始
    if current_best_reward is None:
        current_best_reward = _run_data(init_pra)
        current_best_pra = copy.deepcopy(init_pra)
        save_best_params(current_best_reward, current_best_pra, filename)
        print(f"Initialized best: reward={current_best_reward}, params={current_best_pra}")
    
    print(f"Current best reward: {current_best_reward}, params: {current_best_pra}")
    
    # 运行MCTS搜索
    new_params, new_reward = mcts_search(init_pra, iterations, simulation_depth, c)
    
    print(f"This search best reward: {new_reward}, params: {new_params}")
    
    # 检查是否更新
    if new_reward > current_best_reward:
        current_best_reward = new_reward
        current_best_pra = copy.deepcopy(new_params)
        save_best_params(current_best_reward, current_best_pra, filename)
        print(f"Updated global best! New best reward: {current_best_reward}, params: {current_best_pra}")
    else:
        print("No update to global best.")
    
    return new_params, new_reward, current_best_pra, current_best_reward

# 示例使用：多次调用
if __name__ == "__main__":
    # 第一次调用
    init_pra1 = [48, 52, 52, 152, 48, 48, 68, 164, 80, 76]
    new_params1, new_reward1, global_pra1, global_reward1 = run_mcts_with_update(init_pra1, iterations=5000)
    
