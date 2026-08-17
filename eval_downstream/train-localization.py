
import torch
from torch import optim
try:
    from unet_utils.tensorboard_visualizer import TensorboardVisualizer
except ModuleNotFoundError:
    TensorboardVisualizer = None
from unet_utils.loss import FocalLoss, SSIM
import os
import random
from unet_utils.data_loader import MVTec_Anomaly_Detection, MVTecDRAEMTestDataset_partial
from torch.utils.data import DataLoader
import numpy as np
import csv
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
except ModuleNotFoundError:
    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(bool)
        y_score = np.asarray(y_score, dtype=float)
        pos = int(y_true.sum())
        neg = int((~y_true).sum())
        if pos == 0 or neg == 0:
            return float("nan")

        order = np.argsort(y_score, kind="mergesort")
        ranks = np.empty(len(y_score), dtype=float)
        sorted_scores = y_score[order]
        rank = 1
        i = 0
        while i < len(sorted_scores):
            j = i + 1
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            avg_rank = (rank + rank + (j - i) - 1) / 2.0
            ranks[order[i:j]] = avg_rank
            rank += j - i
            i = j
        return float((ranks[y_true].sum() - pos * (pos + 1) / 2.0) / (pos * neg))

    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true).astype(bool)
        y_score = np.asarray(y_score, dtype=float)
        pos = int(y_true.sum())
        if pos == 0:
            return 0.0

        order = np.argsort(-y_score, kind="mergesort")
        y_true = y_true[order]
        y_score = y_score[order]
        distinct = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        precision = tps / (tps + fps)
        recall = tps / pos
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def precision_recall_curve(y_true, y_score):
        y_true = np.asarray(y_true).astype(bool)
        y_score = np.asarray(y_score, dtype=float)
        pos = int(y_true.sum())
        if y_true.size == 0:
            return np.array([1.0]), np.array([0.0]), np.array([], dtype=float)
        if pos == 0:
            thresholds = np.unique(y_score)[::-1]
            return np.ones(len(thresholds) + 1), np.zeros(len(thresholds) + 1), thresholds

        order = np.argsort(-y_score, kind="mergesort")
        y_true = y_true[order]
        y_score = y_score[order]
        distinct = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs].astype(float)
        fps = (1 + threshold_idxs - tps).astype(float)
        precisions = tps / np.maximum(tps + fps, 1.0)
        recalls = tps / pos
        thresholds = y_score[threshold_idxs]
        return np.r_[precisions, 1.0], np.r_[recalls, 0.0], thresholds
from unet_utils.model_unet import DiscriminativeSubNetwork
import os
from unet_utils.au_pro_util import calculate_au_pro

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def max_f1_from_pr_curve(y_true, y_score):
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    denom = precisions + recalls
    f1_scores = np.divide(
        2 * precisions * recalls,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    finite_scores = f1_scores[np.isfinite(f1_scores)]
    return float(finite_scores.max()) if finite_scores.size else float("nan")


def max_f1_from_threshold_sweep(y_true, y_score):
    gt = np.asarray(y_true).astype(bool)
    scores = np.asarray(y_score, dtype=float)
    eps = 1e-8
    best_f1 = float("-inf")
    for threshold in np.arange(0.0, 1.0 + 1e-3, 0.01):
        pred = scores > threshold
        intersect = np.logical_and(gt, pred).sum()
        pred_area = pred.sum()
        gt_area = gt.sum()
        precision = intersect / (pred_area + eps)
        recall = intersect / (gt_area + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        best_f1 = max(best_f1, float(f1))
    return best_f1 if np.isfinite(best_f1) else float("nan")


def selection_score(metrics, selection_metric="pixel"):
    if selection_metric == "legacy":
        values = [
            metrics["AUC Image"],
            metrics["AUC Pixel"],
            metrics["AP Pixel"],
            metrics["PRO Pixel"],
        ]
    elif selection_metric == "all":
        values = [
            metrics["AUC Image"],
            metrics["AP Image"],
            metrics["F1 Image"],
            metrics["AUC Pixel"],
            metrics["AP Pixel"],
            metrics["F1 Pixel"],
            metrics["PRO Pixel"],
        ]
    elif selection_metric == "pixel_ap":
        values = [metrics["AP Pixel"]]
    elif selection_metric == "pixel_f1":
        values = [metrics["F1 Pixel"]]
    elif selection_metric == "pixel":
        values = [
            metrics["AUC Pixel"],
            metrics["AP Pixel"],
            metrics["F1 Pixel"],
            metrics["PRO Pixel"],
        ]
    else:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("-inf")
    return float(sum(finite_values))


RESULT_FIELDS = [
    "dataset",
    "obj",
    "task",
    "split",
    "best_epoch",
    "selection_metric",
    "acc",
    "auc_image",
    "ap_image",
    "f1_image",
    "auc_pixel",
    "ap_pixel",
    "f1_pixel",
    "pro_pixel",
    "checkpoint",
]


def init_result_csv(path):
    if not path:
        return
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def append_result_csv(path, row):
    if not path:
        return
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULT_FIELDS)
        writer.writerow(row)


def result_row(args, obj_name, task, split, best_epoch, metrics, checkpoint):
    return {
        "dataset": args.dataset_name,
        "obj": obj_name,
        "task": task,
        "split": split,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "acc": "",
        "auc_image": metrics["AUC Image"],
        "ap_image": metrics["AP Image"],
        "f1_image": metrics["F1 Image"],
        "auc_pixel": metrics["AUC Pixel"],
        "ap_pixel": metrics["AP Pixel"],
        "f1_pixel": metrics["F1 Pixel"],
        "pro_pixel": metrics["PRO Pixel"],
        "checkpoint": checkpoint,
    }


def test(args, obj_name, model_seg, device, split='all', label='test'):
    mvtec_path = args.mvtec_path
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    img_dim = 256
    model_seg.eval()
    dataset = MVTecDRAEMTestDataset_partial(mvtec_path +'/' + obj_name + "/test/", resize_shape=[img_dim, img_dim], split=split)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    gt_masks=[]
    predicted_masks=[]

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].to(device)
        gray_batch=gray_batch[:,[2,1,0],:,:]

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        out_mask = model_seg(gray_batch)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        gt_masks.append(true_mask_cv.squeeze())
        predicted_masks.append(out_mask_cv.squeeze())

        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
    f1_image = max_f1_from_pr_curve(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    f1_pixel = max_f1_from_threshold_sweep(total_gt_pixel_scores, total_pixel_scores)
    pro_pixel, _ = calculate_au_pro(gt_masks, predicted_masks)
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
    print(obj_name + " " + label + " split=" + split + " n=" + str(len(dataset)))
    print("AUC Image:  " +str(auroc))
    print("AP Image:  " +str(ap))
    print("F1 Image:  " +str(f1_image))
    print("AUC Pixel:  " +str(auroc_pixel))
    #print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print("F1 Pixel:  " +str(f1_pixel))
    print('PRO Pixel:' +str(pro_pixel))
    print("==============================")
    return {
        "AUC Image": float(auroc),
        "AP Image": float(ap),
        "F1 Image": float(f1_image),
        "AUC Pixel": float(auroc_pixel),
        "AP Pixel": float(ap_pixel),
        "F1 Pixel": float(f1_pixel),
        "PRO Pixel": float(pro_pixel),
    }


def train_on_device(obj_names, args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    init_result_csv(args.result_csv)

    for obj_name in obj_names:

        run_name = obj_name

        model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
        model_seg.to(device)
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_focal = FocalLoss()

        dataset = MVTec_Anomaly_Detection(args,obj_name,length=500)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16,
                                worker_init_fn=seed_worker, generator=generator)

        n_iter = 0
        last_sum=float("-inf")
        best_epoch = None
        best_metrics = None
        for epoch in range(args.epochs):
            model_seg.train()
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                aug_gray_batch = sample_batched["image"].to(device)
                anomaly_mask = sample_batched["mask"].to(device)
                out_mask = model_seg(aug_gray_batch)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = segment_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter +=1
            scheduler.step()

            metrics = test(args, obj_name, model_seg, device, split=args.selection_split, label='selection')
            sum_metric = selection_score(metrics, args.selection_metric)
            if sum_metric>last_sum:
                torch.save(model_seg.state_dict(), os.path.join(args.save_path, run_name + ".pckl"))
                last_sum=sum_metric
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                best_metrics["Selection Score"] = sum_metric
                print("New best selection metrics")
                print("Selection Split: " + args.selection_split)
                print("Selection Metric: " + args.selection_metric)
                print("Best Epoch: " + str(best_epoch))
                for name, value in best_metrics.items():
                    print(name + ": " + str(value))
                print("==============================")

        if best_metrics is None:
            print("No selection metrics were produced. Check --epochs; it may be 0.")
        else:
            best_checkpoint = os.path.join(args.save_path, run_name + ".pckl")
            print("Best selection metrics for " + obj_name)
            print("Selection Split: " + args.selection_split)
            print("Selection Metric: " + args.selection_metric)
            print("Best Epoch: " + str(best_epoch))
            for name, value in best_metrics.items():
                print(name + ": " + str(value))
            print("Best checkpoint: " + best_checkpoint)
            print("==============================")
            append_result_csv(
                args.result_csv,
                result_row(
                    args,
                    obj_name,
                    "localization_selection_best",
                    args.selection_split,
                    best_epoch,
                    best_metrics,
                    best_checkpoint,
                ),
            )

            if args.final_test_split != 'none':
                model_seg.load_state_dict(torch.load(best_checkpoint, map_location=device))
                final_metrics = test(args, obj_name, model_seg, device, split=args.final_test_split, label='final')
                final_metrics["Selection Score"] = selection_score(final_metrics, args.selection_metric)
                print("Final held-out test metrics for " + obj_name)
                print("Final Test Split: " + args.final_test_split)
                print("Selected Epoch: " + str(best_epoch))
                for name, value in final_metrics.items():
                    print(name + ": " + str(value))
                print("==============================")
                append_result_csv(
                    args.result_csv,
                    result_row(
                        args,
                        obj_name,
                        "localization_final",
                        args.final_test_split,
                        best_epoch,
                        final_metrics,
                        best_checkpoint,
                    ),
                )

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_name', type=str, default='all')
    parser.add_argument('--generated_data_path', action='store', type=str, required=True)
    parser.add_argument('--save_path', default='checkpoints/localization', type=str)
    parser.add_argument('--mask_root', action='store', type=str, required=True)
    parser.add_argument('--mvtec_path', action='store', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int,default=8, required=False)
    parser.add_argument('--lr', action='store', type=float,default=0.0001, required=False)
    parser.add_argument('--epochs', action='store', type=int,default=200, required=False)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--seed', action='store', type=int, default=2026, required=False)
    parser.add_argument('--selection_split', choices=['all', 'val', 'test'], default='val')
    parser.add_argument('--final_test_split', choices=['all', 'val', 'test', 'none'], default='test')
    parser.add_argument('--selection_metric', choices=['pixel', 'pixel_ap', 'pixel_f1', 'legacy', 'all'], default='pixel')
    parser.add_argument('--result_csv', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--extra_real_anomaly_prob', action='store', type=float, default=0.1)
    parser.add_argument('--log_path', action='store', type=str,default='./logs/ ', required=False)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_separately', action='store_true',default=False)
    parser.add_argument('--reverse', action='store_true',default=False)
    parser.add_argument('--data_name',type=str, default='text_inversion')
    
    args = parser.parse_args()
    set_seed(args.seed)

    obj_batch =  [
                    'bottle',
                    'capsule',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
    if args.reverse:
        obj_batch=reversed(obj_batch)
    if args.sample_name!='all':
        obj_list=[args.sample_name]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    if torch.cuda.is_available():
        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)
    else:
        train_on_device(picked_classes, args)
#python train-unet.py --data_path $path_to_the_generated_data  --save_path ./ --mvtec_path=$path_to_mvtec --sample_name=capsule
