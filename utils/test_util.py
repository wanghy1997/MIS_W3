import os.path
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm
from skimage.measure import label
from scipy.ndimage import zoom
# from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
from datetime import datetime
currentDateAndTime = datetime.now()

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def validation_all_case_MMWHS(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16, save_nii_dir=None):
    loader = tqdm(image_list)
    total_metric = []
    
    if save_nii_dir is not None:
            save_nii_dir = save_nii_dir+ '/results'
            os.makedirs(save_nii_dir, exist_ok=True)
    for case_idx in loader:
        # image_path = base_dir + '/btcv_h5/{}.h5'.format(case_idx)
        # h5f = h5py.File(image_path, 'r')
        # image, gt_mask = h5f['image'][:], h5f['label'][:]
        image_path = base_dir + '/npy/ct_train_{}_image.npy'.format(case_idx)  # 0001_image.npy
        label_path = base_dir + '/npy/ct_train_{}_label.npy'.format(case_idx)
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            raise ValueError(case_idx)
        image = np.load(image_path)
        gt_mask = np.load(label_path)
        prediction, score_map = test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # ================= 保存 nii.gz（可视化用） =================
        if save_nii_dir is not None:
            save_npy_path = os.path.join(save_nii_dir, f'{case_idx}_pred.npy')
            np.save(save_npy_path, prediction.astype(np.uint8))

        prediction = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(prediction, size=(160, 160, 80) ,mode='nearest').int()
        prediction = prediction.squeeze().numpy().astype(np.int8)       
        gt_mask = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = F.interpolate(gt_mask, size=(160, 160, 80) ,mode='nearest').int()
        gt_mask = gt_mask.squeeze().numpy().astype(np.int8)
        if np.sum(prediction) == 0:
            case_metric = np.zeros((4, num_classes - 1))
        else:
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric

    
def validation_all_case_btcv(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16, save_nii_dir=None):
    loader = tqdm(image_list)
    total_metric = []
    if save_nii_dir is not None:
        save_nii_dir = save_nii_dir+ '/results'
        os.makedirs(save_nii_dir, exist_ok=True)
    for case_idx in loader:
        image_path = base_dir + '/btcv_h5/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]
        # image_path = base_dir + '/{}_image.npy'.format(case_idx)  # 0001_image.npy
        # label_path = base_dir + '/{}_label.npy'.format(case_idx)
        # if not os.path.exists(image_path) or not os.path.exists(label_path):
        #     raise ValueError(case_idx)
        # image = np.load(image_path)
        # gt_mask = np.load(label_path)
        # prediction, score_map = test_single_case_btcv2d_cpc(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        prediction, score_map = test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        # ================= 保存 nii.gz（可视化用） =================``````
        if save_nii_dir is not None:
            save_npy_path = os.path.join(save_nii_dir, f'{case_idx}_pred.npy')
            np.save(save_npy_path, prediction.astype(np.uint8))
        prediction = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(prediction, size=(160, 160, 80) ,mode='nearest').int()
        prediction = prediction.squeeze().numpy().astype(np.int8)       
        gt_mask = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = F.interpolate(gt_mask, size=(160, 160, 80) ,mode='nearest').int()
        gt_mask = gt_mask.squeeze().numpy().astype(np.int8)
        if np.sum(prediction) == 0:
            case_metric = np.zeros((4, num_classes - 1))
        else:
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def validation_all_case_btcv_2d(
    model,
    num_classes,
    base_dir,
    image_list,
    image_size=128,
    multimask_output=True,
    promptmode='point',
    branch='branch2',
    save_nii_dir=None,
):
    loader = tqdm(image_list)
    total_metric = []

    if save_nii_dir is not None:
        save_nii_dir = save_nii_dir + '/results'
        os.makedirs(save_nii_dir, exist_ok=True)

    for case_idx in loader:
        image_path = base_dir + '/btcv_h5/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]   # [D,H,W]

        prediction, score_map = test_single_case_btcv2d_cpc(
            model=model,
            image=image,
            image_size=image_size,
            num_classes=num_classes,
            multimask_output=multimask_output,
            promptmode=promptmode,
            branch=branch,
        )

        if save_nii_dir is not None:
            save_npy_path = os.path.join(save_nii_dir, f'{case_idx}_pred.npy')
            np.save(save_npy_path, prediction.astype(np.uint8))

        prediction_t = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
        prediction_t = F.interpolate(prediction_t, size=(160, 160, 80), mode='nearest').int()
        prediction = prediction_t.squeeze().numpy().astype(np.int8)

        gt_mask_t = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask_t = F.interpolate(gt_mask_t, size=(160, 160, 80), mode='nearest').int()
        gt_mask = gt_mask_t.squeeze().numpy().astype(np.int8)

        case_metric = np.zeros((4, num_classes - 1))
        if np.sum(prediction) != 0:
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)

        total_metric.append(np.expand_dims(case_metric, axis=0))

    all_metric = np.concatenate(total_metric, axis=0)   # [N,4,C-1]
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric

    
def validation_all_case(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16):
    loader = tqdm(image_list)
    total_metric = []
    for case_idx in loader:
        image_path = base_dir + '/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]
        prediction, score_map = test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        prediction = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(prediction, size=(160, 160, 80) ,mode='nearest').int()
        prediction = prediction.squeeze().numpy().astype(np.int8)       
        gt_mask = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = F.interpolate(gt_mask, size=(160, 160, 80) ,mode='nearest').int()
        gt_mask = gt_mask.squeeze().numpy().astype(np.int8)
        if np.sum(prediction) == 0:
            case_metric = np.zeros((4, num_classes - 1))
        else:
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def validation_all_case_flare(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16):
    loader = tqdm(image_list)
    total_metric = []
    for case_idx in loader:
        image_path = base_dir + '/npy/{}_image.npy'.format(case_idx)  # 0001_image.npy
        label_path = base_dir + '/npy/{}_label.npy'.format(case_idx)
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            raise ValueError(case_idx)
        image = np.load(image_path)
        gt_mask = np.load(label_path)
        prediction, score_map = test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        prediction = torch.FloatTensor(prediction).unsqueeze(0).unsqueeze(0)
        # to maintain consistency with the size in DHC, https://github.com/xmed-lab/DHC/blob/main/code/data/preprocess_amos.py
        prediction = F.interpolate(prediction, size=(160, 160, 80) ,mode='nearest').int()
        prediction = prediction.squeeze().numpy().astype(np.int8)       
        gt_mask = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0)
        gt_mask = F.interpolate(gt_mask, size=(160, 160, 80) ,mode='nearest').int()
        gt_mask = gt_mask.squeeze().numpy().astype(np.int8)
        
        if np.sum(prediction) == 0:
            case_metric = np.zeros((4, num_classes - 1))
        else:
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8

    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def validation_all_case_fast(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16):
    loader = tqdm(image_list)
    total_metric = []
    for case_idx in loader:
        image_path = base_dir + '/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]
        prediction, score_map = test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            case_metric = np.zeros((1, num_classes - 1))
        else:
            case_metric = np.zeros((1, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric_dice(prediction == i, gt_mask == i)
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8
    all_metric = np.concatenate(total_metric, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode='constant',
            constant_values=0
        )

    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for x in range(sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(sz):
                    zs = min(stride_z * z, dd - patch_size[2])

                    test_patch = image[
                        xs:xs + patch_size[0],
                        ys:ys + patch_size[1],
                        zs:zs + patch_size[2]
                    ].astype(np.float32)

                    test_patch = torch.from_numpy(test_patch).unsqueeze(0).unsqueeze(0).cuda()

                    outputs = model(test_patch)[0]
                    outputs = F.softmax(outputs, dim=1)
                    outputs = outputs.squeeze(0).cpu().numpy()  # [C, D, H, W]

                    score_map[
                        :,
                        xs:xs + patch_size[0],
                        ys:ys + patch_size[1],
                        zs:zs + patch_size[2]
                    ] += outputs

                    cnt[
                        xs:xs + patch_size[0],
                        ys:ys + patch_size[1],
                        zs:zs + patch_size[2]
                    ] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[
            wl_pad:wl_pad + w,
            hl_pad:hl_pad + h,
            dl_pad:dl_pad + d
        ]
        score_map = score_map[
            :,
            wl_pad:wl_pad + w,
            hl_pad:hl_pad + h,
            dl_pad:dl_pad + d
        ]

    return label_map, score_map


# def test_single_case_fast(model, image, stride_xy, stride_z, patch_size, num_classes=1):
#     w, h, d = image.shape
#     #print(w, h, d)

#     # if the size of image is less than patch_size, then padding it
#     add_pad = False
#     if w < patch_size[0]:
#         w_pad = patch_size[0] - w
#         add_pad = True
#     else:
#         w_pad = 0
#     if h < patch_size[1]:
#         h_pad = patch_size[1] - h
#         add_pad = True
#     else:
#         h_pad = 0
#     if d < patch_size[2]:
#         d_pad = patch_size[2] - d
#         add_pad = True
#     else:
#         d_pad = 0
#     wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
#     hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
#     dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
#     if add_pad:
#         image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
#     ww, hh, dd = image.shape
    
    

#     sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
#     sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
#     sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
#     # print("{}, {}, {}".format(sx, sy, sz))
#     # score_map : CxDxHxW, cnt: DxHxW
#     score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
#     cnt = np.zeros(image.shape).astype(np.float32)    
    
#     bs = 0
#     total_bs = 0
#     image = torch.from_numpy(image.astype(np.float32)).cuda() 
#     for x in range(0, sx):
#         xs = min(stride_xy * x, ww - patch_size[0])
#         for y in range(0, sy):
#             ys = min(stride_xy * y, hh - patch_size[1])
#             for z in range(0, sz):
#                 zs = min(stride_z * z, dd - patch_size[2])
#                 test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
#                 test_patch = test_patch.unsqueeze(0).unsqueeze(0)
                
#                 if bs==0:
#                     test_patches = test_patch
#                     test_patches_info = {str(bs): (xs, ys, zs)}
#                 else:
#                     test_patches = torch.cat((test_patches, test_patch), dim=0)
#                     test_patches_info[str(bs)] = (xs, ys, zs)

#                 cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
#                     = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1                  
                
#                 bs += 1
#                 total_bs += 1
#                 if bs == 8 or total_bs == sx * sy * sz:
#                     with torch.no_grad():
#                         # outputs = model(test_patches)[0]
#                         outputs = model(test_patches)
#                         # if len(outputs) > 1:
#                         #     outputs = outputs[0]            # torch.Size([bs, 14, 96, 96, 96])
#                         outputs = F.softmax(outputs, dim=1) # torch.Size([bs, 14, 96, 96, 96])  
#                     outputs = outputs.cpu().data.numpy()            
#                     for i in range(bs):
#                         output_score = outputs[i, :, :, :, :]
#                         xs, ys, zs = test_patches_info[str(i)]
#                         score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
#                         = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + output_score
#                     bs = 0
                    
#     # score map: CxDxHxW, label_map: CxDxHxW -> DxHxW
#     score_map = score_map / np.expand_dims(cnt, axis=0)
#     label_map = np.argmax(score_map, axis=0)
#     if add_pad:
#         label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
#         score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
#     return label_map, score_map

def test_single_case_btcv2d(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # score_map : CxDxHxW, cnt: DxHxW
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)[0]
                    # if len(y) > 1:
                    #     y = y[0]
                    y = F.softmax(y, dim=1)

                y = y.squeeze(0).cpu().data.numpy()
                # y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    # score map: CxDxHxW, label_map: CxDxHxW -> DxHxW
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def test_single_case_btcv2d_cpc(
    model,
    image,                      # numpy [D, H, W]
    image_size=128,
    num_classes=14,
    multimask_output=True,
    promptmode='point',
    branch='branch2',           # 'branch1' or 'branch2'
):
    """
    输入:
        image: [D, H, W] 的 3D numpy volume

    输出:
        prediction: [D, H, W]
        score_map:  [num_classes, D, H, W]
    """
    model.eval()

    D, H, W = image.shape
    prediction = np.zeros((D, H, W), dtype=np.uint8)
    score_map = np.zeros((num_classes, D, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(D):
            slice_2d = image[z]   # [H, W]
            x, y = slice_2d.shape

            # resize 到训练时的 2D 输入尺寸
            slice_resized = zoom(slice_2d, (image_size / x, image_size / y), order=0)

            input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
            # 如果你的模型要求 3 通道，就打开下面这句
            # input_tensor = input_tensor.repeat(1, 3, 1, 1)

            outputs = model(input_tensor, multimask_output, image_size, -1, promptmode)

            if branch == 'branch1':
                logits = outputs['low_res_logits1']   # [1, C, h, w]
            else:
                logits = outputs['low_res_logits2']   # [1, C, h, w]

            probs = torch.softmax(logits, dim=1)

            # 上采样回 128x128（若当前 low_res 不是 image_size）
            probs_up = F.interpolate(
                probs,
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            )

            pred_slice = torch.argmax(probs_up, dim=1).squeeze(0).cpu().numpy()   # [128,128]

            # resize 回原始 slice 尺寸
            pred_slice = zoom(pred_slice, (x / image_size, y / image_size), order=0)
            prediction[z] = pred_slice.astype(np.uint8)

            probs_np = probs_up.squeeze(0).cpu().numpy()  # [C, 128, 128]
            for c in range(num_classes):
                score_map[c, z] = zoom(
                    probs_np[c],
                    (x / image_size, y / image_size),
                    order=1
                )

    return prediction, score_map


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # score_map : CxDxHxW, cnt: DxHxW
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)[0]
                    # if len(y) > 1:
                    #     y = y[0]
                    y = F.softmax(y, dim=1)

                y = y.squeeze(0).cpu().data.numpy()
                # y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    # score map: CxDxHxW, label_map: CxDxHxW -> DxHxW
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map

def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        # sf = compute_surface_distances(gt, pred, spacing_mm=(1., 1., 1.))
        # nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
        ji = metric.binary.jc(pred, gt)
    elif pred.sum() == 0 and gt.sum() > 0:
        dice = 0
        hd95 = 128
        asd = 128
        ji = 0
    elif pred.sum() == 0 and gt.sum() == 0:
        dice = 1
        hd95 = 0
        asd = 0
        ji = 1
    elif pred.sum() > 0 and gt.sum() == 0:
        dice = 0
        hd95 = 128
        asd = 128
        ji = 0
    return np.array([dice, ji, hd95, asd])

def cal_metric_dice(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        return np.array([dice])
    else:
        return np.zeros(1)

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        total_dice[i - 1] += metric.binary.dc(prediction_tmp, label_tmp)

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def dice_ratio(prediction, label, num=2):
    """
    dice ratio
    :param masks:
    :param labels:
    :return:
    """
    masks = prediction.cpu().data.numpy()
    labels = label.cpu().data.numpy()
    bs = masks.shape[0]
    total_dice = np.zeros(num - 1)
    for i in range(bs):
        case_dice_tmp = cal_dice(masks[i, :], labels[i, :], num)
        total_dice += case_dice_tmp

    return total_dice / bs


