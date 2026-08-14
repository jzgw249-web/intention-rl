# 任务03修订版 GATE 报告：最小可行实验

运行日期：2026-08-14  
环境：`MiniGrid-Empty-8x8-v0`  
训练：4条件 × 8种子（42–49）× 100,000步  
评估：固定30个episode seed，裸环境，`deterministic=False`为主指标  
评估间隔：10,000步

## 实际改动

- 新增 `intention_pb`：`lambda * (gamma * Phi(s') - Phi(s))`，
  `Phi=(cos+1)/2`，默认 `lambda=1.0`。
- `gamma` 由 `train.py` 的单一常量传入PPO和wrapper。
- `terminated` 时后继势取0；`truncated` 时保留实际后继势，双边界测试通过。
- 训练环境保留相应reward wrapper；评估环境完全不套reward wrapper。
- `success_rate`、`native_return`、`episode_length` 分别写入三个CSV。
- 随机策略为主指标，确定性策略保留为附属列；所有条件使用相同评估seed组。
- `intention` 保留为 `intention_naive` 的兼容别名。
- 批量脚本最大并发默认4；README、requirements、gitignore和视频脚本完成修复。
- `intention_wrapper.py`、`dense_wrapper.py`、`results_summary.txt` 未修改。

## 主实验结果（mean +/- std，n=8训练种子）

| 条件 | 首次达到50%成功率 | 最终成功率 | 最终原生回报 | 最终episode长度 |
|---|---:|---:|---:|---:|
| sparse | 12,500 +/- 4,330步 | 1.000 +/- 0.000 | 0.955 +/- 0.010 | 12.68 +/- 2.71 |
| intention_pb | 28,750 +/- 7,806步 | 1.000 +/- 0.000 | 0.947 +/- 0.025 | 14.94 +/- 7.25 |
| dense | 17,500 +/- 4,330步 | 1.000 +/- 0.000 | 0.959 +/- 0.003 | 11.73 +/- 0.76 |
| intention_naive | 8/8未达到 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 256.00 +/- 0.00 |

所有最终值取PPO完整rollout后的最后评估点（约100,352步）。

## H1：intention_pb相对sparse是否加速

按首次达到50%成功率的平均步数计算：

`acceleration = sparse_steps / intention_pb_steps = 12,500 / 28,750 = 0.435x`

因此没有加速；`intention_pb`达到同一阈值所需步数约为sparse的2.30倍。
本轮结果不支持“directional intention提高样本效率”的H1。

## intention_pb vs intention_naive

- `intention_naive`：8/8最终成功率0，episode全部跑满256步。
- `intention_pb`：8/8均达到50%阈值，最终成功率1.00，平均episode约15步。

这直接复现并修复了形式错误：恒正即时奖励改变行为目标，potential-based形式则恢复了任务完成能力。不过，“恢复可学习”不等于“比无塑形更快”。

## 天花板效应

Sparse平均仅12.5k步达到50%，评估分辨率又只有10k，因此环境存在强烈天花板效应：

- 加速空间很小；
- 首次50%基本只能落在10k或20k等离散检查点；
- 100k时sparse、PB和dense最终成功率全部饱和到1.00。

尽管如此，PB比sparse慢约16.25k步，方向明确，不是被天花板掩盖的小幅正加速。
若后续要检验更细的样本效率差异，应由人决定是否换到更难环境；本轮没有换环境。

## 简报中考虑不周之处

1. 默认 `lambda=1.0` 未做预注册尺度依据。potential形式保证的是底层MDP的回报等价结构，不保证有限样本PPO更容易优化；当前PB较慢可能包含尺度导致的优化负担。
2. Empty-8x8的起点、朝向和目标全部固定。30个评估seed不是30个不同任务实例，只是随机策略动作采样的30次Monte Carlo rollout。
3. `ImgObsWrapper`不向策略显式提供全局朝向，而势函数使用 `agent_dir`。这是原方案的历史设定，本轮为最小可比性没有修改，但理论表述需要注明策略观测与塑形信息并不完全一致。
4. `dense`使用未归一化的逐步负距离，和其他条件奖励尺度不同，不是严格意义上的公平阳性对照。本轮按要求保留原实现。
5. 10k评估间隔使“首次50%”精度较粗；当前均值应理解为区间化测量，不是精确到单步的样本复杂度。

## 产物

- 原始评估数据：`main_logs/`
- 32个最终模型：`main_models/`
- 曲线和统计表：`main_results/`
- 原地转圈演示：`gate1_diagnostics/intention_loitering.mp4`

历史错误结论仍完整保留在 `results_summary.txt`。
