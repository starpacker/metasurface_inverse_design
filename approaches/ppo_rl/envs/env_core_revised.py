import numpy as np
import os
import random
import copy
from envs import package as pkg
import time,sys
import importlib.util
import pickle
import subprocess

flag_env = 2
if flag_env == 2:    # no judge
    from envs.collect_data_2 import get_reward
    pra_txt = "C:\\data\\pra_all.txt"
    reward_txt = "C:\\data\\reward.txt"
elif flag_env == 5:  # multi_EP no judge
    from envs.multi_collect_data import get_reward
    pra_txt = "C:\\data\\pra_all_5.txt"
    reward_txt = "C:\\data\\reward_5.txt"


def boundary_test(pra):
    c = pra[0]>=30 and pra[0]<=160 and pra[2]>=40 and pra[2]<=80 \
    and pra[3]>=80 and pra[3]<=340 and pra[0]+pra[3]<=370 \
    and pra[6]>=30 and pra[6]<=150 and pra[7]>=40 and pra[7]<=340 \
    and pra[6]+pra[7]<=340 and pra[9]>=30 and pra[9]<=100 \
    and pra[8]>=40 and pra[8] <=100 and pra[5]>=40 and pra[5]<=100 \
    and pra[4]>=40 and pra[4]<=100 and pra[1]>=40 and pra[1]<=100 \
    and pra[9]+pra[8]+pra[5]+pra[4]+pra[1]<=370
    return c

def round_to_nearest_multiple_of_four(value):
    # 将值四舍五入到最近的4的整倍数
    return np.round(value / 4) * 4

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, tunable_indices=None, fixed_values=None):
        """
        初始化环境
        :param tunable_indices: 可调参数的索引列表，例如 [0, 1, 6, 7]
        :param fixed_values: 固定参数的值列表，长度为10，不可调参数的固定值，None表示使用上一次的值
        """
        self.best_reward = -np.inf
        self.best_pra = None
        self.reward_call_count = 0
        # self.search_triggered = False
        self.search_triggered = True
        self.search_threshold = 200

        # [112,48,52,152,48,48,112,92,80,76]
        # self.agent_num = 4
        self.agent_num = 1

        self.tunable_indices =  [0, 1, 6, 7]  # 默认可调参数索引
        self.fixed_values = [None,None,52,152,48,48,None,None,80,76]
        # self.fixed_values = [None,None,60,144,60,48,None,None,72,76]
        
        # self.tunable_indices =  [0, 1, 4, 6, 7]  # 默认可调参数索引
        # self.fixed_values = [None,None,52,152,None,48,None,None,80,76]
        # self.tunable_indices = [1,7,8]
        # self.fixed_values = [112,None,52,152,60,48,112,None,None,76]
        # # baseline
        # self.tunable_indices = [0,1,2,3,4,5,6,7,8]
        # self.fixed_values = [None,None,None,None,None,None,None,None,None,50]

        self.obs_dim = len(self.tunable_indices)  # 观测维度等于可调参数数量
        self.action_dim = len(self.tunable_indices)  # 动作维度等于可调参数数量
        
        # 参数边界定义
        self.param_bounds = {
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

    def reset(self):
        """
        重置环境
        """
        full_obs = self._generate_full_obs()

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_agent_obs.append(self._normalize_selected(full_obs))
        self.last_obs = full_obs
        self.dones = [False for i in range(self.agent_num)]
        return sub_agent_obs

    def _generate_full_obs(self):
        """
        生成符合边界条件的完整参数数组
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            obs = [0] * 10
            
            # 设置可调参数
            for idx in self.tunable_indices:
                if idx == 0:
                    obs[idx] = random.randint(8, 40) * 4  # [32, 160]
                elif idx == 1:
                    obs[idx] = random.randint(10, 25) * 4  # [40, 100]
                elif idx == 2:
                    obs[idx] = random.randint(10, 20) * 4  # [40, 80]
                elif idx == 3:
                    obs[idx] = random.randint(20, (370 - obs[0]) // 4) * 4
                elif idx == 4:
                    obs[idx] = random.randint(10, 25) * 4
                elif idx == 5:
                    obs[idx] = random.randint(10, 25) * 4
                elif idx == 6:
                    obs[idx] = random.randint(8, 37) * 4  # [32, 148]
                elif idx == 7:
                    obs[idx] = random.randint(10, (340 - obs[6]) // 4) * 4
                elif idx == 8:
                    obs[idx] = random.randint(10, 25) * 4
                elif idx == 9:
                    obs[idx] = random.randint(8, 25) * 4  # [32, 100]
            
            # 设置固定参数
            for i in range(10):
                if i not in self.tunable_indices:
                    if self.fixed_values[i] is not None:
                        obs[i] = self.fixed_values[i]
                    else:
                        # 使用上一次的值，如果是第一次则使用默认值
                        if hasattr(self, 'last_obs') and self.last_obs is not None:
                            obs[i] = self.last_obs[i]
                        else:
                            # 使用边界中值作为默认值
                            min_val, max_val = self.param_bounds[i]
                            obs[i] = (min_val + max_val) // 2
            
            if boundary_test(obs):
                return obs
        
        # 如果随机生成失败，使用一个已知有效的参数组合
        return [112, 48, 52, 152, 48, 48, 112, 92, 80, 76]

    def _normalize_selected(self, obs_full):
        """
        只对指定索引的参数进行归一化
        """
        selected_obs = [obs_full[i] for i in self.tunable_indices]
        normalized = []

        for i, idx in enumerate(self.tunable_indices):
            min_val, max_val = self.param_bounds[idx]
            norm_val = (selected_obs[i] - min_val) / (max_val - min_val)
            normalized.append(norm_val)
        return np.array(normalized)

    def _denormalize_selected(self, obs_norm):
        """
        只对指定索引的参数进行反归一化
        """
        denormalized = []

        for i, idx in enumerate(self.tunable_indices):
            min_val, max_val = self.param_bounds[idx]
            denorm_val = obs_norm[i] * (max_val - min_val) + min_val
            denormalized.append(denorm_val)
        return np.array(denormalized)

    def read_data(self):
        with open('my_class.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        return loaded_data

    def to_pra(self, actions):
        actions = np.argmax(actions, axis=1)
        pras = []
        for action in actions:
            b = (action + 8) * 4
            pras.append(b)
        return np.array(pras)

    def step(self, actions):
        """
        执行一步动作
        """
        if self.agent_num == 1:
            actions = actions[0]

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        # 反归一化得到可调参数
        denorm_actions = self._denormalize_selected(actions)

        # 构建完整参数数组
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            full_obs = copy.copy(self.last_obs)  # 上一次状态作为基础
        else:
            full_obs = [0] * 10

        for i, idx in enumerate(self.tunable_indices):
            full_obs[idx] = int(round_to_nearest_multiple_of_four(denorm_actions[i]))

        # 设置固定参数（如果需要重新设置）
        for i in range(10):
            if i not in self.tunable_indices and self.fixed_values[i] is not None:
                full_obs[i] = self.fixed_values[i]

        # 边界检查
        if not boundary_test(full_obs):
            full_obs = [0] * 10  # 不合法时清零
            reward = self._get_reward(full_obs)
            for i in range(self.agent_num):
                sub_agent_obs.append(self._normalize_selected(full_obs))
                sub_agent_reward.append(reward)
                sub_agent_done.append(False)
                sub_agent_info.append({})
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

        reward = self._get_reward(full_obs)
        self.last_obs = full_obs
        
        for i in range(self.agent_num):
            sub_agent_obs.append(self._normalize_selected(full_obs))
            sub_agent_reward.append(reward)
            sub_agent_done.append(False)
            sub_agent_info.append({})
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def _get_reward(self, pras_int):
        """
        获取奖励值
        """
        self.reward_call_count += 1
        np.savetxt(pra_txt, pras_int)
        get_reward()
        
        reward = np.loadtxt(reward_txt)

        # 更新全局最优
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_pra = pras_int.copy()

        # 检查是否触发搜索
        if self.search_triggered and self.reward_call_count % self.search_threshold == 0:
            print(f"\n[INFO] Reward call count reached {self.search_threshold}. Triggering parameter search...")
            self.run_parameter_search()
            time.sleep(10)
        return reward
    
    def run_parameter_search(self):
        if self.best_pra is None:
            initial =[ 48,  52,  52 ,152,  48 , 48 , 68 ,164,  80,  76]
        else:
            initial = self.best_pra.copy()

        # from envs.best_response import run_best_response_search
        # from envs.best_response_4 import run_best_response_search
        from envs.best_response_1 import run_best_response_search
        result = run_best_response_search(initial, self.tunable_indices,max_simulations_per_param=4)
        # decrease the best-response part 

        # 更新环境状态
        self.last_obs = result["final_pra"]
        if self.best_reward < result["best_reward"]:
            self.best_reward = result["best_reward"]
            self.best_pra = result["final_pra"].copy()
        print(self.reward_call_count, self.best_reward,self.best_pra)

        print("[INFO] Best Response search completed. Environment updated.")

    def run_mcts_search(self):
        if self.best_pra is None:
            initial =[ 48,  52,  52 ,152,  48 , 48 , 68 ,164,  80,  76]
        else:
            initial = self.best_pra.copy()

        from envs.MCTS_response import run_mcts_response_search
        result = run_mcts_response_search(initial, self.tunable_indices)

        # 更新环境状态
        self.last_obs = result["final_pra"]
        self.best_reward = result["best_reward"]
        self.best_pra = result["final_pra"].copy()
        print(self.reward_call_count, self.best_reward,self.best_pra)

        print("[INFO] MCTS search completed. Environment updated.")