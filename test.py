# test.py
import torch
import os
import json
import argparse
import utils
from logger import Logger
from sac_ae import SacAeAgent
from envs.env_wrappers import DummyVecEnv

def load_args(work_dir):
    with open(os.path.join(work_dir, "args.json"), "r") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

def make_test_env(args):
    from envs.env_continuous import ContinuousActionEnv
    def get_env_fn():
        def init_env():
            env = ContinuousActionEnv()
            env.seed(args.seed)
            return env
        return init_env

    return DummyVecEnv([get_env_fn()])

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

def load_model(agent, model_dir, step):
    agent.actor.load_state_dict(
        torch.load(os.path.join(model_dir, f"actor_{step}.pt"))
    )
    agent.critic.load_state_dict(
        torch.load(os.path.join(model_dir, f"critic_{step}.pt"))
    )
    print(f"Loaded model from step {step}")

def main():
    work_dir = "./log"          # 你的训练目录
    model_step = 500000         # 你要测试的 checkpoint
    obs_dim_global = 9
    action_dim_global = 9

    args = load_args(work_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    utils.set_seed_everywhere(args.seed)

    env = make_test_env(args)

    agent = make_agent(
        obs_shape=obs_dim_global,
        action_shape=action_dim_global,
        args=args,
        device=device
    )

    model_dir = os.path.join(work_dir, "model")
    load_model(agent, model_dir, model_step)

    # 关闭训练模式
    agent.actor.eval()
    agent.critic.eval()

    L = Logger(work_dir, use_tb=False)

    evaluate(
        env=env,
        agent=agent,
        video=None,
        num_episodes=10,
        L=L,
        step=model_step
    )

if __name__ == "__main__":
    main()
