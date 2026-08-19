#!/bin/bash
# λ 扫描：FourRooms 上的剂量—反应曲线。
# λ=0 与 λ=1 两个端点已由 task-05 提供（sparse 与 potential_geo），此处只补四个中间/外推值。
set -u
for LAM in 0.1 0.25 0.5 2.0; do
  echo "===== lambda = $LAM ====="
  .venv312/Scripts/python.exe run_experiments.py \
    --conditions potential_geo \
    --seeds 42 43 44 45 46 47 48 49 \
    --env-id MiniGrid-FourRooms-v0 \
    --shaping-coeff "$LAM" \
    --total-timesteps 500000 \
    --eval-freq 10000 \
    --eval-episodes 30 \
    --eval-seed-base 30000 \
    --max-workers 4 \
    --log-dir lambda_logs --save-dir lambda_models --stdout-dir lambda_stdout \
    || { echo "lambda $LAM FAILED"; exit 1; }
done
echo "ALL LAMBDA DONE"
