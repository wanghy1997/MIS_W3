from medpy import metric
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn.functional as F


def getLargestCC(segmentation):
    from skimage.measure import label
    labels = label(segmentation)
    if labels.max() != 0:
        largestCC = labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)
    else:
        largestCC = segmentation
    return largestCC


def calculate_metric_percase(pred, gt):
    """
    pred, gt: binary numpy array
    return: [DSC, HD95, JI, ASD]
    """
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        ji = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return [dice, hd95, ji, asd]

    if pred.sum() == 0 and gt.sum() == 0:
        return [1.0, 0.0, 1.0, 0.0]

    return [0.0, 0.0, 0.0, 0.0]


def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


def infer_single_case_btcv2d_dual(args, image, sam_model, SGDL, num_classes, eval_model='sgdl'):
    """
    输入:
        image: numpy array, [D, H, W]
    输出:
        prediction: [D, H, W]
        score_map:   [C, D, H, W]  (可用于保留原接口)
    """
    patch_size = [args.image_size, args.image_size]
    SGDL.load_state_dict(torch.load(args.save_best_path, weights_only=True))
    SGDL.eval()
    prediction = np.zeros_like(image, dtype=np.uint8)
    score_map = np.zeros((num_classes, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)

    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice_2d = image[ind, :, :]
            x, y = slice_2d.shape[0], slice_2d.shape[1]
            # resize 到网络输入尺寸
            slice_resized = zoom(slice_2d, (patch_size[0] / x, patch_size[1] / y), order=0)
            input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
            input_tensor = input_tensor.repeat(1, 3, 1, 1)   # [1,3,H,W]

            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL(input_tensor)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)   # [1,C,H,W]
            out = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().numpy()  # [H,W]
            # resize 回原 slice 尺寸
            pred_slice = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred_slice.astype(np.uint8)
            # 保存 score_map，便于保持原接口
            final_soft_np = fusion_map_soft.squeeze(0).cpu().numpy()  # [C,H,W]
            for c in range(num_classes):
                score_map[c, ind] = zoom(
                    final_soft_np[c],
                    (x / patch_size[0], y / patch_size[1]),
                    order=1
                )

    return prediction, score_map


def test_single_volume(args, image, label, sam_model, SGDL, keep_largest_cc=False):
    """
    返回:
        metric_array_sam:  shape [4, classes-1]
        metric_array_sgdl: shape [4, classes-1]

    顺序统一为:
        [DSC, HD95, JI, ASD]
    """
    classes = args.num_classes
    patch_size = [args.image_size, args.image_size]

    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    # 如果还有多余 channel，去掉
    if image.ndim == 4:
        image = image[0]
    if label.ndim == 4:
        label = label[0]

    sam_prediction = np.zeros_like(label, dtype=np.uint8)
    sgdl_prediction = np.zeros_like(label, dtype=np.uint8)

    sam_model.eval()
    SGDL.eval()

    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice_2d = image[ind, :, :]
            x, y = slice_2d.shape[0], slice_2d.shape[1]

            slice_resized = zoom(
                slice_2d,
                (patch_size[0] / x, patch_size[1] / y),
                order=0
            )

            input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
            input_tensor = input_tensor.repeat(1, 3, 1, 1)

            # SGDL forward
            pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL(input_tensor)
            fusion_map_soft = torch.softmax(fusion_map, dim=1)

            # SGDL 最终多类预测
            out_sgdl = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().detach().numpy()
            pred_sgdl_resized = zoom(out_sgdl, (x / patch_size[0], y / patch_size[1]), order=0)
            sgdl_prediction[ind] = pred_sgdl_resized.astype(np.uint8)

            # SAM forward（基于 SGDL 提供的 prompt 信息）
            image_embeddings = sam_model.image_encoder(input_tensor)
            points_embedding, boxes_embedding, mask_embedding = sam_model.super_prompt(image_embeddings)

            low_res_mask_list = []
            mask_prompt_size = sam_model.prompt_encoder.mask_input_size

            for i in range(args.num_classes):
                mask_prompt = F.interpolate(
                    fusion_map[:, i, ...].unsqueeze(1).clone().detach(),
                    size=mask_prompt_size,
                    mode='bilinear',
                    align_corners=False,
                )

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes_embedding[i],
                    masks=mask_prompt,
                )

                low_res_masks, iou_predictions = sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=args.multimask,
                )

                low_res_mask_list.append(low_res_masks)

            low_res_masks_all = torch.cat(low_res_mask_list, dim=1)  # [1, C, h, w]

            pred_sam = F.interpolate(
                low_res_masks_all,
                size=(args.image_size, args.image_size),
                mode='bilinear',
                align_corners=False,
            )
            pred_sam_soft = torch.softmax(pred_sam, dim=1)

            out_sam = torch.argmax(pred_sam_soft, dim=1).squeeze(0).cpu().detach().numpy()
            pred_sam_resized = zoom(out_sam, (x / patch_size[0], y / patch_size[1]), order=0)
            sam_prediction[ind] = pred_sam_resized.astype(np.uint8)

    # 分别计算 SAM 与 SGDL 的四个指标
    metric_list_sam = []
    metric_list_sgdl = []

    for i in range(1, classes):
        pred_sam_i = (sam_prediction == i).astype(np.uint8)
        pred_sgdl_i = (sgdl_prediction == i).astype(np.uint8)
        gt_i = (label == i).astype(np.uint8)

        if keep_largest_cc:
            if pred_sam_i.sum() > 0:
                pred_sam_i = getLargestCC(pred_sam_i)
            if pred_sgdl_i.sum() > 0:
                pred_sgdl_i = getLargestCC(pred_sgdl_i)

        metric_list_sam.append(calculate_metric_percase(pred_sam_i, gt_i))
        metric_list_sgdl.append(calculate_metric_percase(pred_sgdl_i, gt_i))

    # [classes-1, 4] -> [4, classes-1]
    metric_array_sam = np.array(metric_list_sam, dtype=np.float32).T
    metric_array_sgdl = np.array(metric_list_sgdl, dtype=np.float32).T

    return metric_array_sam, metric_array_sgdl


if __name__ == '__main__':
    import cv2
    import torch
    import argparse

    import torch.nn.functional as F
    from dataloader.dataset import build_Dataset
    from dataloader.transforms import build_transforms
    from torch.utils.data import DataLoader
    import numpy as np

    from Model.model import KnowSAM

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='./SampleData',
                        help='Name of Experiment')
    parser.add_argument('--dataset', type=str, default='/ACDC',
                        help='Name of Experiment')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=list, default=256,
                        help='patch size of network input')
    parser.add_argument('--point_nums', type=int, default=5, help='points number')
    parser.add_argument('--box_nums', type=int, default=1, help='boxes number')
    parser.add_argument('--mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument('--thd', type=bool, default=False, help='3d or not')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")

    parser.add_argument('--sam_model_path', type=str,
                        default="./sam_best_model.pth",
                        help='model weight path')
    parser.add_argument('--SGDL_model_path', type=str,
                        default="./SGDL_iter_16400.pth",
                        help='model weight path')

    args = parser.parse_args()
    data_transforms = build_transforms(args)

    test_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="test_list",
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = "SGDL"
    if model == "SGDL":
        SGDL_model = KnowSAM(args, bilinear=True).to(args.device).train()
        SGDL_checkpoint = torch.load(args.SGDL_model_path)
        SGDL_model.load_state_dict(SGDL_checkpoint)
        SGDL_model.eval()

        avg_dice_list = 0.0
        avg_iou_list = 0.0
        avg_hd95_list = 0.0
        avg_asd_list = 0.0
        classes = args.num_classes
        patch_size = [args.image_size, args.image_size]
        final_res = [0, 0, 0, 0, 0]

        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            image, label = test_image.squeeze(0).cpu().detach().numpy(), test_label.squeeze(0).cpu().detach().numpy()
            SGDL_prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                input = input.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    pred_UNet, pred_VNet, pred_UNet_soft, pred_VNet_soft, fusion_map = SGDL_model(input)
                    fusion_map_soft = torch.softmax(fusion_map, dim=1)
                    out_SGDL = torch.argmax(fusion_map_soft, dim=1).squeeze(0).cpu().detach().numpy()
                    pred_SGDL = zoom(out_SGDL, (x / patch_size[0], y / patch_size[1]), order=0)
                    SGDL_prediction[ind] = pred_SGDL

            metric_list = []
            for i in range(1, classes):
                disc_pred = SGDL_prediction == i
                gt = label == i
                disc_pred[disc_pred > 0] = 1
                if 1:
                    disc_pred = getLargestCC(disc_pred)
                gt[gt > 0] = 1
                single_class_res = []
                if disc_pred.sum() > 0:
                    single_class_res.append(metric.binary.dc(disc_pred, gt))
                    single_class_res.append(metric.binary.jc(disc_pred, gt))
                    single_class_res.append(metric.binary.asd(disc_pred, gt))
                    single_class_res.append(metric.binary.hd95(disc_pred, gt))
                else:
                    single_class_res = [0, 0, 0, 0, 0]
                metric_list.append(single_class_res)

            metric_list = np.array(metric_list).astype("float32")
            metric_list = np.mean(metric_list, axis=0)

            print(metric_list)
            final_res += metric_list
        final_res = [x / len(test_loader) for x in final_res]
        print("avg_dice: ", final_res[0])
        print("avg_iou: ", final_res[1])
        print("avg_asd: ", final_res[2])
        print("avg_hd95: ", final_res[3])




