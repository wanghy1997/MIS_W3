#!/bin/bash

# set -e 作用：如果任何一行命令报错（比如显存溢出），脚本会立即停止，不再运行后续命令。
# 如果你希望不管报错与否都继续跑下一个，请删除下面这一行。
set -e

echo "========== 开始运行命令 1 (Label 10%, Baseline) =========="
CUDA_VISIBLE_DEVICES=1 python train_AMOS_Baseline.py --seed 1337 --label_num 10 --remark "Baseline"
# CUDA_VISIBLE_DEVICES=1 python train_AMOS_test04.py --seed 1337 --label_num 50 --remark "text_0"
# CUDA_VISIBLE_DEVICES=0 python train_BTCV_Ours.py --seed 1337 --label_num 50 --remark "text_0"
echo "========== 命令 1 完成，休息 10 秒冷却 GPU... =========="
sleep 10

echo "========== 开始运行命令 2 (Label 50%, Baseline) =========="
CUDA_VISIBLE_DEVICES=1 python train_AMOS_Baseline.py --seed 1337 --label_num 50 --remark "Baseline"

echo "========== 所有实验运行完毕！ =========="