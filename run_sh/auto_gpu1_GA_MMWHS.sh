#!/bin/bash

# set -e 作用：如果任何一行命令报错（比如显存溢出），脚本会立即停止，不再运行后续命令。
# 如果你希望不管报错与否都继续跑下一个，请删除下面这一行。
set -e

echo "========== 开始运行命令 1 (Label 10, Ours) =========="
CUDA_VISIBLE_DEVICES=1 python train_MMWHS_GA.py --seed 1337 --label_num 10 --remark "baseline"
echo "========== 命令 1 完成，休息 10 秒冷却 GPU... =========="
sleep 10

echo "========== 开始运行命令 2 (Label 50, Ours) =========="
CUDA_VISIBLE_DEVICES=1 python train_MMWHS_GA.py --seed 1337 --label_num 50 --remark "baseline"
echo "========== 命令 2 完成，休息 10 秒冷却 GPU... =========="

echo "========== 所有实验运行完毕！ =========="