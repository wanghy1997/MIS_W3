import sys
import numpy as np
import torch

sys.argv = ["train_MMWHS_GA.py", *sys.argv[1:]]
import train_MMWHS_GA as base


def main():
    base.os.makedirs(base.snapshot_path, exist_ok=True)
    metric_all_cases, best_model_path = base.train(base.labeled_list, base.unlabeled_list)
    model = base.create_model(n_classes=base.num_classes, cube_size=base.cube_size, patch_size=base.patch_size[0])
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    _, _, metric_final = base.test_util.validation_all_case_MMWHS(
        model,
        num_classes=base.num_classes,
        base_dir=base.train_data_path,
        image_list=base.test_list,
        patch_size=base.patch_size,
        stride_xy=24,
        stride_z=16,
        save_nii_dir=base.snapshot_path,
    )
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)
    metric_log_path = base.os.path.join(base.snapshot_path, f"metric_final_{base.args.dataset_name}_{base.args.exp}.npy")
    np.save(metric_log_path, metric_final)
    print(f"BEST_CHECKPOINT={best_model_path}")
    print(f"FINAL_AVG_DSC={metric_mean[0].mean():.4f}")
    print(f"FINAL_AVG_JI={metric_mean[1].mean():.4f}")
    print(f"FINAL_AVG_HD95={metric_mean[2].mean():.4f}")
    print(f"FINAL_AVG_ASD={metric_mean[3].mean():.4f}")


if __name__ == "__main__":
    main()
