import os
import sys
import stat
import numpy as np
import torch

argv = sys.argv[1:]
if "--use_medsam2" not in argv:
    argv += ["--use_medsam2", "1"]
if "--exp" not in argv:
    argv += ["--exp", "SAM2SSL_fromscratch"]
sys.argv = ["work3_AMOS_Baseline.py", *argv]
import work3_AMOS_Baseline as base


def main():
    os.makedirs(base.snapshot_path, exist_ok=True)
    os.chmod(base.snapshot_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
    metric_final, best_model_path = base.train(base.labeled_list, base.unlabeled_list, base.eval_list)
    model = base.create_model(n_classes=base.num_classes, cube_size=base.cube_size, patchsize=base.patch_size[0])
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    _, _, metric_final = base.test_amos.validation_all_case(
        model,
        num_classes=base.num_classes,
        base_dir=base.train_data_path,
        image_list=base.test_list,
        patch_size=base.patch_size,
        stride_xy=32,
        stride_z=16,
    )
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)
    metric_save_path = os.path.join(base.snapshot_path, f"metric_final_{base.args.dataset_name}_{base.args.exp}.npy")
    np.save(metric_save_path, metric_final)
    handler, sh = base.config_log(base.snapshot_path, "total_metric")
    base.logging.info(
        "Final Average DSC:{:.4f}, HD95: {:.4f}, NSD: {:.4f}, ASD: {:.4f}".format(
            metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(), metric_mean[3].mean()
        )
    )
    base.logging.getLogger().removeHandler(handler)
    base.logging.getLogger().removeHandler(sh)
    print(f"BEST_CHECKPOINT={best_model_path}")
    print(f"FINAL_AVG_DSC={metric_mean[0].mean():.4f}")


if __name__ == "__main__":
    main()
