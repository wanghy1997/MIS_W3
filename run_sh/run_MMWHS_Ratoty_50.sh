#!/bin/bash

# set -e 作用：如果任何一行命令报错（比如显存溢出），脚本会立即停止，不再运行后续命令。
# 如果你希望不管报错与否都继续跑下一个，请删除下面这一行。

echo "========== 开始运行命令 0 (Label 50 MMWHS_Ours) =========="
CUDA_VISIBLE_DEVICES=1 python train_MMWHS_Ours.py --seed 1337 --label_num 50 --remark "Ratoty" 
echo "========== 命令 0 完成，休息 10 秒冷却 GPU... =========="
sleep 10

# echo "========== 开始运行命令 3 (Label 50, MMWHS_AblationStudy_3) =========="
CUDA_VISIBLE_DEVICES=1 python train_MMWHS_Ours_3.py --seed 1337 --label_num 50 --remark "Ab_3_Ratoty" --max_epoch 5000 --start_contrast_epoch 10

# echo "========== 命令 3 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

echo "========== 开始运行命令 5 (Label 50, MMWHS_AblationStudy_5) =========="
CUDA_VISIBLE_DEVICES=1 python train_MMWHS_Ours_5.py --seed 1337 --label_num 50 --remark "Ab_5_Ratoty" --max_epoch 5000 --start_contrast_epoch 10

echo "========== 所有实验运行完毕！ =========="