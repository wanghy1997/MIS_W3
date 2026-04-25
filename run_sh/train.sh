# Baseline
CUDA_VISIBLE_DEVICES=0 python train_AMOS_MagicNet_GA_TAK.py --seed 1337 --label_num 10 --remark "BioCLIP"

# single3090 用于提升 Baseline 性能
# 次卡包含两张卡，故使用 CUDA_VISIBLE_DEVICES=【0】
# amos ours  5%
CUDA_VISIBLE_DEVICES=0 python train_AMOS_MagicNet_GA_ours.py --seed 1337 --label_num 10 --remark "ema2"

# amos ours  2%
CUDA_VISIBLE_DEVICES=0 python train_AMOS_MagicNet_GA_ours.py --seed 1337 --label_num 4 --remark "ema1"


# double3090 用于验证改进方法的有效性
# 次卡包含两张卡，故使用 CUDA_VISIBLE_DEVICES=【0，1】
# amos ours  5%
CUDA_VISIBLE_DEVICES=0 python train_AMOS_MagicNet_GA_ours.py --seed 1337 --label_num 10 --remark "fuse_new_test"

# amos ours  2%
CUDA_VISIBLE_DEVICES=1 python train_AMOS_MagicNet_GA_ours.py --seed 1337 --label_num 4 --remark "ema2"


# test01
CUDA_VISIBLE_DEVICES=1 python train_AMOS_test01.py --seed 1337 --label_num 10 --remark "hyper2param"

# 
CUDA_VISIBLE_DEVICES=0 python train_AMOS_test04.py --seed 1337 --label_num 10 --remark "semint1"
CUDA_VISIBLE_DEVICES=1 python train_AMOS_test04.py --seed 1337 --label_num 10 --remark "Ratoty"
CUDA_VISIBLE_DEVICES=0 python train_AMOS_test04.py --seed 1337 --label_num 50 --remark "Ratoty"

# BTCV 
CUDA_VISIBLE_DEVICES=0 python train_BTCV_GA.py --label_num 30 --remark "baseline"
CUDA_VISIBLE_DEVICES=0 python train_BTCV_GA.py --label_num 40 --remark "baseline"
CUDA_VISIBLE_DEVICES=0 python train_BTCV_Ours.py --label_num 30 --remark "ours"
CUDA_VISIBLE_DEVICES=0 python train_BTCV_Ours.py --label_num 40 --remark "ours"

# FLARE
CUDA_VISIBLE_DEVICES=0 python train_FLARE_GA.py --label_num 10 --remark "baseline"
CUDA_VISIBLE_DEVICES=0 python train_FLARE_Ours.py --label_num 10 --remark "ours"
