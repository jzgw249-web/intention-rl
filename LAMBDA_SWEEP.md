# λ 扫描：怎么跑

给负责塑形系数扫描的同学。跑完把 `lambda_logs/` 整个打包发回即可。

---

## 1. 跑什么

**环境是 `MiniGrid-FourRooms-v0`，不是 Empty-8x8。**

原因：我们已经在 Empty-8x8 上跑过 8 个种子，距离势函数和基线**基本重合**
（ΔAUC = +0.0033，t = +0.44，5/8 种子）。在一个本来就没有效应的环境里扫 λ，
只会得到"λ 怎么调都一样"，没有信息量。

而在 FourRooms 上塑形**明确有害**（ΔAUC = −0.0669，8/8 种子同号）。
那里扫 λ 才能回答真正的问题：

- **λ → 0 时是否收敛回基线？** —— 正确性检查，证明危害确实来自塑形项本身
- **危害是否随 λ 单调增长？** —— 支持"塑形注入的是有害方差"这个机制解释

---

## 2. 只需要跑 4 个值

两个端点已经有了，不用重复：

| λ | 状态 |
|---|---|
| 0 | 已有 —— 就是 `task05_logs/` 里的 `sparse` |
| **0.1** | **要跑** |
| **0.25** | **要跑** |
| **0.5** | **要跑** |
| 1.0 | 已有 —— 就是 `task05_logs/` 里的 `potential_geo` |
| **2.0** | **要跑** |

4 个 λ × 8 个种子 = **32 次训练，约 4.5 小时**，建议夜里跑。

---

## 3. 环境准备

```bash
git clone -b fix/eval-pipeline https://github.com/jzgw249-web/intention-rl.git
cd intention-rl
python -m venv .venv312
.venv312/Scripts/pip install -r requirements.txt      # Linux/Mac: .venv312/bin/pip
```

---

## 4. 命令

把 `--shaping-coeff` 换成 0.1 / 0.25 / 0.5 / 2.0，跑四次：

```bash
.venv312/Scripts/python run_experiments.py \
  --conditions potential_geo \
  --seeds 42 43 44 45 46 47 48 49 \
  --env-id MiniGrid-FourRooms-v0 \
  --shaping-coeff 0.25 \
  --total-timesteps 500000 \
  --eval-freq 10000 \
  --eval-episodes 30 \
  --eval-seed-base 30000 \
  --max-workers 4 \
  --log-dir lambda_logs --save-dir lambda_models --stdout-dir lambda_stdout
```

实验目录名会自动带上 λ（如 `fourrooms_potential_geo_lam0p25_seed42`），
四次跑不会互相覆盖，可以放心重复执行。

### 三个不能改的参数

| 参数 | 为什么 |
|---|---|
| `--eval-seed-base 30000` | 必须与现有的 λ=0 / λ=1 端点一致，否则新旧数据不可比，整轮白跑 |
| `--total-timesteps 500000` | 同上 |
| `--eval-freq 10000` | 同上 |

`--max-workers` 可以按机器调（并发太高会互相抢 CPU，4 比较稳）。

---

## 5. 怎么确认跑对了

启动后看任一条日志：

```bash
tail -3 lambda_stdout/potential_geo_seed42.stdout.log
```

应该看到这样的行：

```
eval@10000: success=0.033, native_return=0.006, episode_length=99.7
```

- `success` 是**裸环境**上的真实成功率（`terminated and not truncated`），
  评估时不套任何 reward wrapper
- 起步接近 0 是正常的，FourRooms 很难，未训练策略成功率就是 0

查进度（数字到 32 就是跑完了）：

```bash
ls lambda_models/*/final_model.zip | wc -l
```

---

## 6. 口径说明

管线已经统一好，用上面的命令就自动对齐，不需要额外设置。列在这里供参考：

- **评估用裸环境**，训练用的 reward wrapper 在评估路径上不会被套上
- **成功判据**是 `terminated and not truncated`
  （MiniGrid 没有 0/1 成功率，其原生奖励是 `1 − 0.9 × 步数 / max_steps`）
- **主指标是 AUC**（成功率曲线下面积除以预算，等于"整个训练过程的平均成功率"），
  不用"首次达到 50% 成功率"——在 FourRooms 上四个条件 32 个种子**没有任何一个达到过 50%**，
  那个指标零信息量
- **主策略是随机策略**（`deterministic=False`）；确定性 argmax 在部分可观测下会退化
- `success_rate` / `native_return` / `episode_length` 分三个 CSV 落盘，不混在一起

---

## 7. 势函数是哪个

`--conditions potential_geo` 对应 `wrappers/potential_distance_wrapper.py`：

```
Φ(s) = −欧氏距离(s, goal) / d_max         值域 [−1, 0]，目标处为 0
F    = λ · [ γ·Φ(s′) − Φ(s) ]
```

**"目标处取 0" 这一条很关键。** 我们实测过：如果势函数在目标处取最大值
（比如 `(cos+1)/2` 那种 `[0,1]` 形式），到达目标那一步的塑形是 `F = 0 − 1.0 = −1.0`，
等于在成功的瞬间给一记最大惩罚。同一环境同一管线的对照：

```
sparse                AUC 0.8934
距离势（[−1, 0]）       AUC 0.8968      ← 与基线持平
cos 势（[0, 1]）       AUC 0.7587      ← 明显更差，8/8 种子同号
```

两者相差 0.138，唯一区别就是势函数在目标处取不取 0。

---

## 8. 跑完之后

把 `lambda_logs/` 整个目录打包发回。模型（`lambda_models/`）不用发，体积大且用不上。
