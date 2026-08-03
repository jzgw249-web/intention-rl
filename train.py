import os
import sys
import argparse
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wrappers.intention_wrapper import IntentionRewardWrapper
from wrappers.dense_wrapper import DenseDistanceWrapper

def make_env(env_id, wrapper_type, seed, log_dir=None):
    def _init():
        env = gym.make(env_id, render_mode=None)
        env.reset(seed=seed)
        
        env = ImgObsWrapper(env)
        
        if wrapper_type == 'intention':
            env = IntentionRewardWrapper(env, coeff=0.5)
        elif wrapper_type == 'dense':
            env = DenseDistanceWrapper(env, coeff=0.1)
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, f"monitor.csv"))
        
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrapper', type=str, choices=['sparse', 'intention', 'dense'], 
                        default='sparse')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        args.total_timesteps = 5000
        print("⚠️ 调试模式：只训练5000步")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = f"{args.wrapper}_seed{args.seed}"
    log_path = os.path.join(args.log_dir, exp_name)
    save_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"🚀 开始训练: {exp_name}")
    print(f"📁 日志: {log_path}")
    print(f"💾 模型: {save_path}")

    env_fn = make_env("MiniGrid-Empty-8x8-v0", args.wrapper, args.seed, log_dir=log_path)
    env = DummyVecEnv([env_fn])

    eval_env_fn = make_env("MiniGrid-Empty-8x8-v0", args.wrapper, args.seed + 1000, log_dir=None)
    eval_env = DummyVecEnv([eval_env_fn])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_path,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=max(2000, args.total_timesteps // 10),
        deterministic=True,
        n_eval_episodes=10,
        render=False,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            tb_log_name=exp_name,
        )
        model.save(os.path.join(save_path, "final_model"))
        print(f"✅ 训练完成！模型保存至 {save_path}")
    except KeyboardInterrupt:
        print("⏹️ 用户中断训练")
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()