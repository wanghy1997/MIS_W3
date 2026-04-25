#!/bin/bash
cd /home/why/SSL4MIS_work3 || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SSL38
export CUDA_VISIBLE_DEVICES=0
python work3_BTCV_Baseline.py   --labelnum 10   --max_iteration 35000   --use_medsam2 1   --medsam2_num_classes 14   --medsam2_warmup 3000   --medsam2_interval 50   --medsam2_cache_size 32   --medsam2_cache_ttl 100000   --medsam2_blend_alpha 0.10   --medsam2_hard_weight 0.10   --medsam2_soft_weight 0.05   --medsam2_prompt_thresh 0.90   --medsam2_teacher_prob_thresh 0.90   --medsam2_min_voxels 1000   --medsam2_min_slice_area 100   --medsam2_max_classes 2   --resume_path /data/why/logs_SAM2SSL/BTCV_SAM2SSL_10labeled/iter_06000_dice_0.548997_best.pth   --resume_iter 6000   --resume_best_dice 0.548997
