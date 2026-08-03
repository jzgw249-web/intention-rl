#!/bin/bash
echo "=== 实验状态检查 ==="
echo ""
echo "📊 训练进程: $(ps aux | grep train.py | grep -v grep | wc -l) / 12 运行中"
echo ""
echo "💾 已保存模型: $(ls models/*/final_model.zip 2>/dev/null | wc -l) / 12"
echo ""
echo "📁 日志文件:"
ls -lh logs/*.log | wc -l
