import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
# import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from sac_ae import SacAeAgent
from envs.env_wrappers import DummyVecEnv

'''
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
'''

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1, type=int)   # default 1000
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"])

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=2,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=2,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=3e4,  # 1e5
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # env parameters
    parser.add_argument("--env_name", type=str, default="MyEnv", help="specify the name of environment")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=True,
        help="Whether to use global state or concatenated obs",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=20, help="Max length for any episode")
    # parser.add_argument("--episode_length", type=int, default=20, help="Max length for any episode")

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=False,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="Whether to use ReLU")
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 5e-4)")

    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,  # 0.2
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.03,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1,  #10.0
        help="max norm of gradients ",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step):
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        # 打印验证模型的平均奖励回报
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()

            # from envs.env_discrete import DiscreteActionEnv
            #
            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 5000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def main():
    obs_dim_global = 4
    action_dim_global = 4
    # obs_dim_global = 9
    # action_dim_global = 9

    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = make_train_env(args)



    # stack several consecutive frames together
    # 对于像素输入的情况，使用帧栈来增加时间维度
    # 这里的帧栈是将连续的帧拼接在一起，形成一个新的观测空间

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    # 限制动作空间的每个维度的数值大小都要在[-1, 1]之间
    # assert env.action_space.low.min() >= -1
    # assert env.action_space.high.max() <= 1

    # 使用的是重放缓冲区
    # 连续帧存放
    # 离散帧抽取进行训练
    replay_buffer = utils.ReplayBuffer(
        obs_shape=obs_dim_global ,
        action_shape=action_dim_global,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    # 创建agent
    agent = make_agent(
        obs_shape= obs_dim_global,
        action_shape=action_dim_global,
        args=args,
        device=device
    )

    # 日志记录器
    L = Logger(args.work_dir, use_tb=args.save_tb)
    
    # episode: 经历了了多少个游戏周期
    # episode_reward: 记录的是一次游戏周期内的总奖励
    # done: 是否结束
    episode = 0
    episode_step, episode_reward, done = 0, 0, False
    start_time = time.time()
    # 限制了总训练的步数
    print("start:",start_time)
    for step in range(args.num_train_steps):
        if step % 1000 == 0:
            print(step)
        if done:
            # 如果游戏结束则进行验证
            if step > 0:
                # 打印训练的时间以及经过的步数
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                # todo 后续查看这个的功能
                L.dump(step)

            # evaluate agent periodically
            if step % args.eval_freq == 0 and step >0:
                # 指定的评估频率
                # 打印评估时的步数
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, args.num_eval_episodes, L, step)
                # 评估后保存模型
                if args.save_model:
                    agent.save(model_dir, step)
                # 可以选择是否保存重放缓冲区，默认不保存
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            # 游戏结束，重置环境
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0 # 游戏周期内经过的步数
            episode += 1

            L.log('train/episode', episode, step)

        # sample action for data collection
        
        if step < args.init_steps:
            # 一开始预热时使用随机采样的动作
            action = env.action_space[0].sample()  # "[0]"
        else:
            # 进入设置为验证模式，退出设置为验证模式
            with utils.eval_mode(agent):
                # 随机动作采样
                # print("random sample")
                action = agent.sample_action(obs)
        # print("action",action)
        # run training update
        if step >= args.init_steps:
            # 如果步数大于预热步数，则进行训练
            # 第一次训练则训练init_steps次，是为了智能体快速利用这些数据进行初步学习
            # 后续每次采样动作后训练一次
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # 对预测的动作进行执行
        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        # 限制游戏周期内的步数
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        if step == 0:
            obs = next_obs

        # 将采集的步数存入重放缓冲区
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
