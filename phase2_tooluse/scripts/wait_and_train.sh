#!/bin/bash
# 等 probe_format_rate.py 跑完（unstable_format_questions.json 生成后），自动启动训练
UNSTABLE="$(dirname "$0")/unstable_format_questions.json"
LOG=/tmp/probe_format.log
TRAIN_LOG=/tmp/train_unstable_fmt.log
PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/rl_study/bin/python

echo "等待筛选完成..."
while [ ! -f "$UNSTABLE" ]; do
    sleep 30
done

echo "筛选完成，开始训练..."
$PYTHON "$(dirname "$0")/train_unstable_format.py" > "$TRAIN_LOG" 2>&1
echo "训练完成，日志: $TRAIN_LOG"
