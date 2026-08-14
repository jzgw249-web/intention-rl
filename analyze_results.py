import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_all_monitor_results(log_dir):
    """加载所有 monitor.csv 文件"""
    data = {}
    for wrapper in ['sparse', 'intention', 'dense']:
        wrapper_data = []
        for seed in [42, 43, 44, 45]:
            # 查找 monitor.csv 文件（可能在子目录中）
            pattern = os.path.join(log_dir, f"{wrapper}_seed{seed}", "**", "monitor.csv")
            files = glob(pattern, recursive=True)
            
            if files:
                # 如果有多个 monitor.csv，选择最大的那个（通常数据最全）
                file = max(files, key=lambda x: os.path.getsize(x))
                try:
                    df = pd.read_csv(file)
                    df['seed'] = seed
                    df['wrapper'] = wrapper
                    wrapper_data.append(df)
                    print(f"✅ 加载: {wrapper}_seed{seed} ({len(df)} 行)")
                except Exception as e:
                    print(f"⚠️ 跳过 {wrapper}_seed{seed}: {e}")
        
        if wrapper_data:
            data[wrapper] = pd.concat(wrapper_data, ignore_index=True)
    return data

def load_evaluations(log_dir):
    """加载 evaluations.npz 文件获取评估结果"""
    data = {}
    for wrapper in ['sparse', 'intention', 'dense']:
        wrapper_data = []
        for seed in [42, 43, 44, 45]:
            pattern = os.path.join(log_dir, f"{wrapper}_seed{seed}", "**", "evaluations.npz")
            files = glob(pattern, recursive=True)
            
            if files:
                file = files[0]
                try:
                    eval_data = np.load(file)
                    # evaluations.npz 包含: 'timesteps', 'results', 'ep_lengths'
                    timesteps = eval_data['timesteps']
                    results = eval_data['results']
                    
                    # 计算每个评估点的平均成功率
                    mean_results = results.mean(axis=1)
                    std_results = results.std(axis=1)
                    
                    for t, mean_r, std_r in zip(timesteps, mean_results, std_results):
                        wrapper_data.append({
                            'seed': seed,
                            'wrapper': wrapper,
                            'total_steps': t,
                            'r': mean_r,
                            'std': std_r
                        })
                    print(f"✅ 加载评估: {wrapper}_seed{seed} ({len(timesteps)} 个评估点)")
                except Exception as e:
                    print(f"⚠️ 跳过评估 {wrapper}_seed{seed}: {e}")
        
        if wrapper_data:
            data[wrapper] = pd.DataFrame(wrapper_data)
    return data

def plot_learning_curves_from_eval(data, save_path="learning_curves.png"):
    """从评估数据绘制学习曲线"""
    plt.figure(figsize=(12, 7))
    
    colors = {'sparse': '#e74c3c', 'intention': '#3498db', 'dense': '#2ecc71'}
    labels = {'sparse': 'Sparse (Baseline)', 'intention': 'Intention', 'dense': 'Dense (Positive Control)'}
    markers = {'sparse': 'o', 'intention': 's', 'dense': '^'}
    
    for wrapper, df in data.items():
        if len(df) == 0:
            continue
        
        # 按步数分组
        df_sorted = df.sort_values('total_steps')
        
        # 计算均值和标准差（按种子）
        grouped = df_sorted.groupby('total_steps').agg({
            'r': ['mean', 'std']
        }).reset_index()
        
        steps = grouped['total_steps'].values
        means = grouped['r']['mean'].values
        stds = grouped['r']['std'].values
        
        # 绘制曲线
        plt.plot(steps, means, color=colors[wrapper], 
                label=labels[wrapper], linewidth=2.5, marker=markers[wrapper], 
                markevery=max(1, len(steps)//10), markersize=8)
        
        # 添加阴影（标准差）
        plt.fill_between(steps, means - stds, means + stds, 
                        color=colors[wrapper], alpha=0.2)

    plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Evaluation Success Rate', fontsize=14, fontweight='bold')
    plt.title('Sample Efficiency Comparison (MiniGrid-Empty-8x8)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 图表保存至 {save_path}")

def compute_stats(data):
    """计算关键统计指标"""
    stats = []
    for wrapper, df in data.items():
        if len(df) == 0:
            continue
        
        # 按种子分组
        seeds = df.groupby('seed')
        
        # 计算每个种子的首次达到50%的步数
        first_50_list = []
        final_success_list = []
        
        for seed, group in seeds:
            group_sorted = group.sort_values('total_steps')
            mask = group_sorted['r'] >= 0.5
            if mask.any():
                first_50 = group_sorted[mask].iloc[0]['total_steps']
                first_50_list.append(first_50)
            
            # 最终成功率（最后20%数据）
            final_df = group_sorted.iloc[-int(max(1, len(group_sorted)*0.2)):]
            final_success_list.append(final_df['r'].mean())
        
        stats.append({
            'wrapper': wrapper,
            'first_50%_steps': f"{np.mean(first_50_list):.0f} ± {np.std(first_50_list):.0f}" if first_50_list else 'N/A',
            'final_success_rate': f"{np.mean(final_success_list):.3f} ± {np.std(final_success_list):.3f}",
            'num_seeds': len(seeds)
        })
    
    stats_df = pd.DataFrame(stats)
    print("\n" + "="*60)
    print("📊 统计结果")
    print("="*60)
    print(stats_df.to_string(index=False))
    print("="*60)
    return stats_df

def main():
    log_dir = "./logs"
    
    print("🔍 正在加载训练数据...\n")
    
    # 尝试从 monitor.csv 加载
    monitor_data = load_all_monitor_results(log_dir)
    
    # 尝试从 evaluations.npz 加载评估数据
    eval_data = load_evaluations(log_dir)
    
    if not eval_data or all(len(df) == 0 for df in eval_data.values()):
        print("\n⚠️ 没有找到评估数据，使用 monitor.csv 数据")
        if monitor_data:
            # 从 monitor.csv 提取评估数据
            plot_data = {}
            for wrapper, df in monitor_data.items():
                # 提取每个 episode 结束时的数据
                episode_data = df[df['l'] > 0].copy()
                if len(episode_data) > 0:
                    episode_data['cum_steps'] = episode_data['total_steps']
                    episode_data['r'] = episode_data['r']
                    plot_data[wrapper] = episode_data
            if plot_data:
                plot_learning_curves_from_monitor(plot_data)
            else:
                print("❌ 没有足够的数据进行绘图")
                return
    else:
        # 使用评估数据绘图
        plot_learning_curves_from_eval(eval_data)
        compute_stats(eval_data)

def plot_learning_curves_from_monitor(data, save_path="learning_curves.png"):
    """从 monitor 数据绘制学习曲线"""
    plt.figure(figsize=(12, 7))
    
    colors = {'sparse': '#e74c3c', 'intention': '#3498db', 'dense': '#2ecc71'}
    labels = {'sparse': 'Sparse (Baseline)', 'intention': 'Intention', 'dense': 'Dense (Positive Control)'}
    
    for wrapper, df in data.items():
        if len(df) == 0:
            continue
        
        # 按步数分组计算滑动平均
        df_sorted = df.sort_values('cum_steps')
        
        # 每 1000 步分组
        df_sorted['bin'] = (df_sorted['cum_steps'] // 1000) * 1000
        grouped = df_sorted.groupby('bin').agg({
            'r': ['mean', 'std']
        }).reset_index()
        
        steps = grouped['bin'].values
        means = grouped['r']['mean'].values
        stds = grouped['r']['std'].values
        
        plt.plot(steps, means, color=colors[wrapper], 
                label=labels[wrapper], linewidth=2.5)
        plt.fill_between(steps, means - stds, means + stds, 
                        color=colors[wrapper], alpha=0.2)

    plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Episode Reward (Smoothed)', fontsize=14, fontweight='bold')
    plt.title('Training Performance Comparison (MiniGrid-Empty-8x8)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 图表保存至 {save_path}")

if __name__ == "__main__":
    main()