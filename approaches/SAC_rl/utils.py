import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    '''
    2. 为什么需要这些处理？
    (1) 量化
    在强化学习中，输入的图像数据通常是高维的（如 84x84x3 的像素值）。
    通过量化，可以减少数据的表示复杂度，同时保留足够的信息。
    (2) 归一化
    归一化到 [0, 1] 的范围可以避免数值过大导致的梯度爆炸问题。
    归一化后的数据更适合输入到神经网络中。
    (3) 添加随机噪声
    添加噪声可以减少量化引入的离散性，使数据更平滑。
    这类似于数据增强的效果，有助于提高模型的泛化能力。
    (4) 中心化
    将数据中心化到 [-0.5, 0.5] 的范围可以让数据的均值接近 0。
    这对使用批归一化（Batch Normalization）或零均值初始化的神经网络非常重要，有助于加速收敛。
    '''
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        # 将图像数据从 8 位（通常是 0-255 的像素值）量化到指定的位数（bits）。
        # 例如，如果 bits=5，则将像素值从 0-255 映射到 0-31 的范围
        obs = torch.floor(obs / 2**(8 - bits))
    # 归一化到 [0, 1]
    obs = obs / bins
    # 添加随机噪声
    # 在归一化后的像素值上添加均匀分布的随机噪声，噪声的范围为 ([0, 1/bins])
    # 这一步是为了避免量化引入的离散性，增加数据的平滑性
    obs = obs + torch.rand_like(obs) / bins
    # 中心化到 [-0.5, 0.5]
    # 这样可以让数据的均值接近 0，有助于神经网络的训练
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """只支持一维向量观测的经验回放缓冲区。"""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = device

        # 统一 shape 为 tuple，并且强制是一维向量
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        else:
            obs_shape = tuple(obs_shape)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)
        else:
            action_shape = tuple(action_shape)

        assert len(obs_shape) == 1,  "This ReplayBuffer only supports 1D vector observations."
        assert len(action_shape) == 1, "This ReplayBuffer only supports 1D vector actions."

        self.obs_dim = obs_shape[0]
        self.act_dim = action_shape[0]

        # 全部 float32（向量观测/动作/奖励/标记）
        self.obses      = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obses = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions    = np.empty((self.capacity, self.act_dim), dtype=np.float32)
        self.rewards    = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones  = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        # 保证写入为 float32 / 标量 → (1,)
        np.copyto(self.obses[self.idx],      np.asarray(obs, dtype=np.float32))
        np.copyto(self.actions[self.idx],    np.asarray(action, dtype=np.float32))
        np.copyto(self.rewards[self.idx],    np.asarray(reward, dtype=np.float32).reshape(1))
        np.copyto(self.next_obses[self.idx], np.asarray(next_obs, dtype=np.float32))
        np.copyto(self.not_dones[self.idx],  np.asarray(1.0 - float(done), dtype=np.float32).reshape(1))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        hi = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, hi, size=self.batch_size)

        obses      = torch.as_tensor(self.obses[idxs],      device=self.device)  # (B, obs_dim)
        actions    = torch.as_tensor(self.actions[idxs],    device=self.device)  # (B, act_dim)
        rewards    = torch.as_tensor(self.rewards[idxs],    device=self.device)  # (B, 1)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device)  # (B, obs_dim)
        not_dones  = torch.as_tensor(self.not_dones[idxs],  device=self.device)  # (B, 1)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{self.last_save}_{self.idx}.pt')
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = sorted(os.listdir(save_dir), key=lambda x: int(x.split('_')[0]))
        for chunk in chunks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end]      = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end]    = payload[2]
            self.rewards[start:end]    = payload[3]
            self.not_dones[start:end]  = payload[4]
            self.idx = end
        self.full = (self.idx == self.capacity)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
