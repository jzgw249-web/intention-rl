#!/usr/bin/env python3
import subprocess
import sys
import os
import time

def run_experiment(wrapper, seed, total_steps=100000):
    cmd = [
        sys.executable,
        "train.py",
        "--wrapper", wrapper,
        "--seed", str(seed),
        "--total_timesteps", str(total_steps),
    ]
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/{wrapper}_seed{seed}.log"
    
    print(f"🚀 启动: {wrapper} seed {seed}")
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return process

if __name__ == "__main__":
    wrappers = ["sparse", "intention", "dense"]
    seeds = [42, 43, 44, 45]
    
    print(f"📊 共 {len(wrappers) * len(seeds)} 个实验")
    print(f"⏱️ 每个约 10万 步")
    
    processes = []
    for wrapper in wrappers:
        for seed in seeds:
            proc = run_experiment(wrapper, seed)
            processes.append(proc)
            time.sleep(0.3)
    
    print("\n✅ 所有实验已启动！")
    print("📝 查看进度: tensorboard --logdir logs")
    print("🔍 查看进程: ps aux | grep train.py")