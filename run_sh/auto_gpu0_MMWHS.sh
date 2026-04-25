#!/bin/bash

# set -e 作用：如果任何一行命令报错（比如显存溢出），脚本会立即停止，不再运行后续命令。
# 如果你希望不管报错与否都继续跑下一个，请删除下面这一行。

# echo "========== 开始运行命令 0 (Label 10, MMWHS_Ours) =========="
# CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours.py --seed 1337 --label_num 10 --remark "from_envl" 
# echo "========== 命令 0 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

# echo "========== 开始运行命令 1 (Label 10, MMWHS_AblationStudy_1) =========="
CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours_1.py --seed 1337 --label_num 10 --remark "AblationStudy_1" --max_epoch 5000 --start_contrast_epoch 10
# echo "========== 命令 1 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

# echo "========== 开始运行命令 2 (Label 10, MMWHS_AblationStudy_2) =========="
CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours_2.py --seed 1337 --label_num 10 --remark "AblationStudy_2" --max_epoch 5000 --start_contrast_epoch 10
# echo "========== 命令 2 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

# echo "========== 开始运行命令 3 (Label 10, MMWHS_AblationStudy_3) =========="
CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours_3.py --seed 1337 --label_num 10 --remark "AblationStudy_3" --max_epoch 5000 --start_contrast_epoch 10

# echo "========== 命令 3 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

# echo "========== 开始运行命令 4 (Label 10, MMWHS_AblationStudy_4) =========="
CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours_4.py --seed 1337 --label_num 10 --remark "AblationStudy_4" --max_epoch 5000 --start_contrast_epoch 10

# echo "========== 命令 4 完成，休息 10 秒冷却 GPU... =========="
# sleep 10

echo "========== 开始运行命令 5 (Label 50, MMWHS_AblationStudy_5) =========="
CUDA_VISIBLE_DEVICES=0 python train_MMWHS_Ours_5.py --seed 1337 --label_num 10 --remark "AblationStudy_5" --max_epoch 5000 --start_contrast_epoch 10

echo "========== 所有实验运行完毕！ =========="