# SAM2 Multiorgan Ablation (2026-04-23)

## Goal

Find which pure-SAM2 prompting strategy is more suitable for multi-organ 3D segmentation on:

- BTCV
- MMWHS
- AMOS

Current implementation:

- per-class prompting
- `F-Mode 4` uniform slice selection
- bidirectional propagation with reset between directions
- single-channel volume converted to 3-channel input

## Existing Results

| Dataset | Variant | Prompt Type | K | RGB | Mean Dice | Mean IoU | Status | Summary |
|---|---|---:|---:|---|---:|---:|---|---|
| BTCV | paper baseline | box | 3 | repeat | 0.3483 | 0.5422 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_repeat_eval_20260422_190855/summary.json` |
| BTCV | more prompts | box | 5 | repeat | 0.3565 | 0.5676 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211712/summary.json` |
| MMWHS | paper baseline | box | 3 | repeat | 0.4695 | 0.4710 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_repeat_eval_20260422_200903/summary.json` |
| MMWHS | more prompts | box | 5 | repeat | 0.5383 | 0.5272 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |
| AMOS | paper baseline | box | 3 | repeat | 0.2329 | 0.2636 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_repeat_eval_20260422_200903/summary.json` |
| AMOS | more prompts | box | 5 | repeat | 0.2566 | 0.2868 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |

## Backbone Scale Check

Fair model-scale comparison keeps the prompt strategy fixed at the strongest pure baseline so far:

- prompt type: `box`
- `K=5`
- `rgb=repeat`
- bidirectional propagation

Only the SAM2 backbone changes:

- `tiny-512`: `sam2.1_hiera_tiny.pt` + `sam2.1_hiera_t512.yaml`
- `large-512`: `sam2.1_hiera_large.pt` + `sam2.1_hiera_l512.yaml`

| Dataset | Variant | Backbone | K | RGB | Mean Dice | Mean IoU | Status | Summary |
|---|---|---|---:|---|---:|---:|---|---|
| BTCV | scale check | tiny-512 | 5 | repeat | 0.3565 | 0.5676 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211712/summary.json` |
| BTCV | scale check | large-512 | 5 | repeat | 0.3676 | 0.5427 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260423_005634/summary.json` |
| MMWHS | scale check | tiny-512 | 5 | repeat | 0.5383 | 0.5272 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |
| MMWHS | scale check | large-512 | 5 | repeat | 0.5282 | 0.5056 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260423_002846/summary.json` |
| AMOS | scale check | tiny-512 | 5 | repeat | 0.2566 | 0.2868 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260422_211531/summary.json` |
| AMOS | scale check | large-512 | 5 | repeat | 0.2355 | 0.2760 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_repeat_eval_20260423_032837/summary.json` |

## Official Checkpoint Family Sweep

This round evaluates the full official checkpoint families exactly as released by Meta:

- `SAM2.1`: `tiny`, `small`, `base_plus`, `large`
- `SAM2`: `tiny`, `small`, `base_plus`, `large`

To keep the prompt strategy fixed at the strongest pure-SAM2 setting found so far, every run uses:

- prompt type: `mask`
- `K=5`
- `rgb=repeat`
- bidirectional propagation
- official backbone config and checkpoint as provided by the model id

This is not the same as the earlier `large-512` fair-scale check. Here we use the direct official model family, which also restores the official input-size choice from each config.

| Dataset | Variant | Family | Model | Prompt Type | K | RGB | Mean Dice | Mean IoU | Status | Summary |
|---|---|---|---|---|---:|---|---:|---:|---|---|
| BTCV | official family sweep | SAM2.1 | tiny | mask | 5 | repeat | 0.4735 | 0.7749 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2.1_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_103952/summary.json` |
| BTCV | official family sweep | SAM2.1 | small | mask | 5 | repeat | 0.4169 | 0.7666 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2.1_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_110404/summary.json` |
| BTCV | official family sweep | SAM2.1 | base_plus | mask | 5 | repeat | 0.4857 | 0.7651 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2.1_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_112957/summary.json` |
| BTCV | official family sweep | SAM2.1 | large | mask | 5 | repeat | 0.4612 | 0.7789 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2.1_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_120327/summary.json` |
| BTCV | official family sweep | SAM2 | tiny | mask | 5 | repeat | 0.6604 | 0.7788 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_125726/summary.json` |
| BTCV | official family sweep | SAM2 | small | mask | 5 | repeat | 0.6217 | 0.7829 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_132129/summary.json` |
| BTCV | official family sweep | SAM2 | base_plus | mask | 5 | repeat | 0.6491 | 0.7786 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_134709/summary.json` |
| BTCV | official family sweep | SAM2 | large | mask | 5 | repeat | 0.5578 | 0.7792 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_142045/summary.json` |
| MMWHS | official family sweep | SAM2.1 | tiny | mask | 5 | repeat | 0.7069 | 0.7189 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2.1_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_103952/summary.json` |
| MMWHS | official family sweep | SAM2.1 | small | mask | 5 | repeat | 0.7172 | 0.7190 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2.1_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_104259/summary.json` |
| MMWHS | official family sweep | SAM2.1 | base_plus | mask | 5 | repeat | 0.7345 | 0.7020 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2.1_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_104635/summary.json` |
| MMWHS | official family sweep | SAM2.1 | large | mask | 5 | repeat | 0.7696 | 0.7334 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2.1_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_105140/summary.json` |
| MMWHS | official family sweep | SAM2 | tiny | mask | 5 | repeat | 0.7758 | 0.7274 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_110045/summary.json` |
| MMWHS | official family sweep | SAM2 | small | mask | 5 | repeat | 0.7307 | 0.6934 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_110433/summary.json` |
| MMWHS | official family sweep | SAM2 | base_plus | mask | 5 | repeat | 0.7431 | 0.6574 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_110839/summary.json` |
| MMWHS | official family sweep | SAM2 | large | mask | 5 | repeat | 0.7485 | 0.7243 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_111414/summary.json` |
| AMOS | official family sweep | SAM2.1 | tiny | mask | 5 | repeat | 0.4588 | 0.5938 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_103952/summary.json` |
| AMOS | official family sweep | SAM2.1 | small | mask | 5 | repeat | 0.4991 | 0.5885 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_120944/summary.json` |
| AMOS | official family sweep | SAM2.1 | base_plus | mask | 5 | repeat | 0.6197 | 0.5960 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_134154/summary.json` |
| AMOS | official family sweep | SAM2.1 | large | mask | 5 | repeat | 0.5020 | 0.6035 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2.1_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_154334/summary.json` |
| AMOS | official family sweep | SAM2 | tiny | mask | 5 | repeat | 0.6154 | 0.6012 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_tiny_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_190149/summary.json` |
| AMOS | official family sweep | SAM2 | small | mask | 5 | repeat | 0.5791 | 0.5943 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_small_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_202727/summary.json` |
| AMOS | official family sweep | SAM2 | base_plus | mask | 5 | repeat | 0.6685 | 0.5992 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_215849/summary.json` |
| AMOS | official family sweep | SAM2 | large | mask | 5 | repeat | 0.4967 | 0.5698 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_sam2_hiera_large_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_235907/summary.json` |

## Overnight Experiments

This round focuses on two questions:

1. Is the main bottleneck prompt type (`point`, `multipoint`, `box`, `mask`)?
2. Do different organs/classes prefer different prompt types?

Class-level preference will be read from each `summary.json`:

- `aggregate.class_mean_dice_present`
- `aggregate.class_mean_iou_nonempty_slices`

### Prompt-Type Sweep

Reference setting is still the strongest pure-SAM2 setting so far:

- `F-Mode 4`
- bidirectional propagation
- per-class prompting

Only the prompt representation or prompt budget changes.

| Dataset | Variant | Prompt Type | K | RGB | Mean Dice | Mean IoU | Status | Summary |
|---|---|---:|---:|---|---:|---:|---|---|
| BTCV | prompt-type sweep | point | 5 | repeat | 0.2875 | 0.3811 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P1_D1_K5_repeat_eval_20260423_001112/summary.json` |
| BTCV | prompt-type sweep | multipoint | 5 | repeat | 0.2753 | 0.3453 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P2_D1_K5_repeat_eval_20260423_001956/summary.json` |
| BTCV | prompt-type sweep | box | 5 | neighbor | 0.3848 | 0.5720 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_K5_neighbor_eval_20260423_002844/summary.json` |
| BTCV | prompt-type sweep | mask | 5 | repeat | 0.3853 | 0.7176 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_003741/summary.json` |
| BTCV | prompt saturation | box | 7 | repeat | 0.3481 | 0.5713 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_paper_best_prompt_F4_P3_D1_K7_repeat_eval_20260423_004645/summary.json` |
| MMWHS | prompt-type sweep | point | 5 | repeat | 0.3404 | 0.3358 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P1_D1_K5_repeat_eval_20260423_001112/summary.json` |
| MMWHS | prompt-type sweep | multipoint | 5 | repeat | 0.2233 | 0.2413 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P2_D1_K5_repeat_eval_20260423_001234/summary.json` |
| MMWHS | prompt-type sweep | box | 5 | neighbor | 0.4911 | 0.5070 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_K5_neighbor_eval_20260423_001357/summary.json` |
| MMWHS | prompt-type sweep | mask | 5 | repeat | 0.6717 | 0.7015 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_001521/summary.json` |
| MMWHS | prompt saturation | box | 7 | repeat | 0.5062 | 0.5161 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_paper_best_prompt_F4_P3_D1_K7_repeat_eval_20260423_001644/summary.json` |
| AMOS | prompt-type sweep | point | 5 | repeat | 0.1240 | 0.1112 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P1_D1_K5_repeat_eval_20260423_001112/summary.json` |
| AMOS | prompt-type sweep | multipoint | 5 | repeat | 0.1183 | 0.1031 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P2_D1_K5_repeat_eval_20260423_005023/summary.json` |
| AMOS | prompt-type sweep | box | 5 | neighbor | 0.2762 | 0.3007 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K5_neighbor_eval_20260423_012921/summary.json` |
| AMOS | prompt-type sweep | mask | 5 | repeat | 0.3764 | 0.5599 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P5_D1_K5_repeat_eval_20260423_020853/summary.json` |
| AMOS | prompt saturation | box | 7 | repeat | 0.2499 | 0.2865 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/AMOS_paper_best_prompt_F4_P3_D1_K7_repeat_eval_20260423_024749/summary.json` |

## Base_Plus Prompt Expansion

This round fixes the backbone to the strongest current official `base_plus` checkpoint on each dataset and asks a more targeted question:

- does increasing prompt density from `K=5` to `K=10` help?
- does sequential prompt mixing (`mask -> box`) help?
- does reusing tracked predictions as new prompts (`feedback`) help?

Current queue settings:

- backbone: `facebook/sam2-hiera-base-plus`
- prompt base: `mask`
- `F-Mode 4`
- bidirectional propagation
- `rgb=repeat`

| Dataset | Variant | Prompt Mix | K | Feedback | Mean Dice | Mean IoU | Status | Summary |
|---|---|---|---:|---|---:|---:|---|---|
| MMWHS | base_plus prompt expansion | mask | 10 | none | 0.8315 | 0.7786 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rnone_FBnone_eval_20260424_012302/summary.json` |
| MMWHS | base_plus prompt expansion | mask -> box | 10 | none | 0.7707 | 0.6725 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBnone_eval_20260424_012856/summary.json` |
| MMWHS | base_plus prompt expansion | mask -> box | 10 | mask feedback | 0.5418 | 0.5074 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/MMWHS_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBmask_eval_20260424_013524/summary.json` |
| BTCV | base_plus prompt expansion | mask | 10 | none | 0.5972 | 0.8276 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rnone_FBnone_eval_20260424_020640/summary.json` |
| BTCV | base_plus prompt expansion | mask -> box | 10 | none | 0.5223 | 0.6825 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBnone_eval_20260424_024446/summary.json` |
| BTCV | base_plus prompt expansion | mask -> box | 10 | mask feedback | 0.4385 | 0.6366 | done | `/data/why/logs_SAM2SSL/medical_sam2_3d_repro/BTCV_sam2_hiera_base_plus_paper_best_prompt_F4_P5_D1_K10_repeat_Rbox_FBmask_eval_20260424_032312/summary.json` |

Note:

- the earlier `BTCV ... Rbox_FBnone_eval_20260424_012127` output only contains `1` case and should be treated as a smoke test, not as a final benchmark row.

## Current Observations

- `BTCV`: prompt type matters, but no single prompt dominates all organs. Overall, `mask` and `box + neighbor` are tied at the top (`0.3853` vs `0.3848`), while `point` and `multipoint` are clearly weaker. Class-wise, `box + neighbor` is strongest for `r.kidney`, `gallbladder`, `liver`, `stomach`, and `inferior vena cava`; `mask` is strongest for `l.kidney`, `portal vein and splenic vein`, and both adrenal glands; `multipoint` is best for `esophagus` and `aorta`; and `box-repeat-K5` still wins `pancreas`. This supports the hypothesis that disjoint abdominal organs do not share one universally best prompt type.
- `MMWHS`: `mask` is decisively best, both overall (`0.6717`) and for every cardiac structure currently measured. `point` and `multipoint` are much weaker, and `box + neighbor` is helpful but still clearly below `mask`. This is consistent with stronger spatial continuity in cardiac anatomy.
- Prompt-budget saturation is not monotonic. On both `BTCV` and `MMWHS`, moving `box repeat` from `K=5` to `K=7` did not improve results.
- `AMOS` shows the same non-monotonic prompt-budget behavior: `box + neighbor + K5` (`0.2762`) is better than `box + K7 repeat` (`0.2499`), so simply adding more box prompts is not enough.
- `SAM2.1 large-512` is not uniformly better. It is slightly better on `BTCV` (`0.3676` vs `0.3565`) but worse on both `MMWHS` (`0.5282` vs `0.5383`) and `AMOS` (`0.2355` vs `0.2566`) under the same prompt setup.
- `AMOS`: the hierarchy is now clear. `point` (`0.1240`) and `multipoint` (`0.1183`) are both weak; `box + neighbor` lifts the mean Dice to `0.2762`; and `mask` improves further to `0.3764`. So AMOS follows the same broad rule as the other datasets: point prompts are weak, and richer spatial prompts are much more effective.
- Official checkpoint family sweep is now complete at `24/24`. The final missing row, `AMOS / SAM2 / large`, ends at `0.4967`, so it does not change the main conclusion: on `AMOS`, the strongest official-family run remains `SAM2 / base_plus` at `0.6685`, and `SAM2` does not scale monotonically with model size there.
- `MMWHS` responds very differently once the backbone is fixed to official `SAM2 base_plus`. Increasing the prompt count from `K=5` to `K=10` with plain `mask` prompting raises mean Dice from `0.7431` to `0.8315`, so this is a strong positive result for denser oracle prompts on the cardiac dataset.
- On the same `MMWHS / base_plus / K=10` setting, sequential `mask -> box` refinement is worse than plain `mask` (`0.7707` vs `0.8315`), and tracked-mask feedback is much worse again (`0.5418`). So the current feedback/self-conditioning implementation is not helping yet; for this dataset it appears to amplify tracker errors rather than stabilize them.
- `BTCV` shows the opposite trend from `MMWHS` under the same official `SAM2 base_plus` backbone. Plain `mask` prompting with `K=10` reaches `0.5972`, which is below the earlier `K=5` official-family result (`0.6491`). So for the abdominal multi-organ setting, simply increasing prompt density is not automatically beneficial and may over-constrain or destabilize the tracker.
- `BTCV` also does not benefit from sequential `mask -> box` refinement at `K=10`. The completed `mask -> box` run falls further to `0.5223`, well below both `BTCV / base_plus / mask / K=10` (`0.5972`) and the earlier official-family `K=5` baseline (`0.6491`).
- `BTCV` feedback/self-conditioning is the worst variant in this round. `mask -> box + feedback(mask)` drops further to `0.4385`, so on `BTCV` the current tracked-prediction recycling strategy is clearly injecting error rather than helping temporal consistency.
- Three-dataset official-family conclusion: `MMWHS` is the clearest case where stronger prompts and official `SAM2` variants can work very well, with `SAM2 / tiny` and `K=10 mask` both strong; `BTCV` prefers the simpler `K=5 mask` official-family setting over denser or mixed prompts; and `AMOS` is dominated more by backbone choice than prompt densification so far, with `SAM2 / base_plus / K=5 mask` still the clean winner.
