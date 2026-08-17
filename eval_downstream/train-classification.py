
import csv
import random
import numpy as np
import torch
from torch import optim
from unet_utils.data_loader import MVTec_classification_train,MVTec_classification_test
from torch.utils.data import DataLoader
import os
import sys
from torchvision.models import ResNet34_Weights, resnet34
import torch.nn as nn
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
def test(args,obj_name, model,anomaly_names,epoch=None):
    model.eval()

    dataset = MVTec_classification_test(args,obj_name,anomaly_names)
    dataloader = DataLoader(dataset, batch_size=100,
                            shuffle=False, num_workers=0)

    total_correct = 0
    total_count = 0
    class_correct = [0 for _ in anomaly_names]
    class_total = [0 for _ in anomaly_names]
    confusion = torch.zeros(len(anomaly_names), len(anomaly_names), dtype=torch.long)

    test_bar = tqdm(
        dataloader,
        desc="Test",
        total=len(dataloader),
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    )

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_bar):
            image, label = sample_batched
            image = image.cuda()
            label = label.cuda()
            y_pred = model(image)
            prediction = torch.argmax(y_pred, 1)
            correct = prediction == label

            total_correct += correct.sum().item()
            total_count += label.numel()
            running_acc = total_correct / total_count if total_count > 0 else 0.0
            test_bar.set_postfix(acc="%.4f" % running_acc)

            for true_label, pred_label in zip(label.cpu(), prediction.cpu()):
                confusion[true_label.item(), pred_label.item()] += 1

            for class_idx in range(len(anomaly_names)):
                class_mask = label == class_idx
                class_count = class_mask.sum().item()
                if class_count == 0:
                    continue
                class_total[class_idx] += class_count
                class_correct[class_idx] += correct[class_mask].sum().item()

    acc = total_correct / total_count if total_count > 0 else 0.0
    print("Accuracy: %.4f (%d/%d)" % (acc, total_correct, total_count), flush=True)
    for class_idx, anomaly_name in enumerate(anomaly_names):
        if class_total[class_idx] == 0:
            print("  %s Accuracy: n/a (0/0)" % anomaly_name, flush=True)
            continue
        class_acc = class_correct[class_idx] / class_total[class_idx]
        print(
            "  %s Accuracy: %.4f (%d/%d)"
            % (anomaly_name, class_acc, class_correct[class_idx], class_total[class_idx]),
            flush=True,
        )

    print_confusion_matrix(confusion, anomaly_names)
    save_confusion_matrix(confusion, anomaly_names, args.checkpoint_path, obj_name, epoch)
    return acc


def print_confusion_matrix(confusion, anomaly_names):
    col_width = max(8, max(len(name) for name in anomaly_names) + 2)
    header = "true\\pred".ljust(col_width) + "".join(name.rjust(col_width) for name in anomaly_names)
    print("Confusion Matrix:", flush=True)
    print(header, flush=True)
    for row_idx, anomaly_name in enumerate(anomaly_names):
        row = anomaly_name.ljust(col_width)
        row += "".join(str(confusion[row_idx, col_idx].item()).rjust(col_width) for col_idx in range(len(anomaly_names)))
        print(row, flush=True)


def save_confusion_matrix(confusion, anomaly_names, checkpoint_path, obj_name, epoch=None):
    os.makedirs(checkpoint_path, exist_ok=True)
    if epoch is None:
        matrix_name = f"{obj_name}_confusion_matrix.csv"
    else:
        matrix_name = f"{obj_name}_confusion_matrix_epoch_{epoch + 1:03d}.csv"
    matrix_path = os.path.join(checkpoint_path, matrix_name)
    with open(matrix_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["true\\pred"] + list(anomaly_names))
        for row_idx, anomaly_name in enumerate(anomaly_names):
            writer.writerow(
                [anomaly_name]
                + [confusion[row_idx, col_idx].item() for col_idx in range(len(anomaly_names))]
            )
    print("Saved confusion matrix: %s" % matrix_path, flush=True)


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


def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    init_result_csv(args.result_csv)

    for obj_name in obj_names:
        print(obj_name, flush=True)
        run_name = obj_name
        dataset = MVTec_classification_train(args,obj_name)
        class_num=dataset.class_num()
        anomaly_names =dataset.return_anomaly_names()
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        model=model.cuda()

        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        criterion = nn.CrossEntropyLoss()
        data_generator = torch.Generator().manual_seed(args.seed)
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16, generator=data_generator)
        max_acc=0
        best_epoch = None
        best_checkpoint = os.path.join(args.checkpoint_path, run_name+".pckl")
        for epoch in range(args.epochs):
            model.train()
            print("Epoch: %d/%d" % (epoch + 1, args.epochs), flush=True)
            epoch_loss = 0.0
            train_bar = tqdm(
                dataloader,
                desc="Train",
                total=len(dataloader),
                dynamic_ncols=True,
                file=sys.stdout,
            )
            for i_batch, sample_batched in enumerate(train_bar):
                image,label=sample_batched
                image=image.cuda()
                label=label.cuda()
                y_pred=model(image)
                loss=criterion(y_pred,label)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                loss_value = loss.item()
                epoch_loss += loss_value
                avg_loss = epoch_loss / (i_batch + 1)
                train_bar.set_postfix(loss="%.6f" % loss_value, avg_loss="%.6f" % avg_loss)

            scheduler.step()
            train_loss = epoch_loss / len(dataloader)
            print("Train Loss: %.6f" % train_loss, flush=True)
            acc = test(args,obj_name, model, anomaly_names, epoch)
            if acc> max_acc:
                max_acc=acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_checkpoint)

        if best_epoch is None:
            print("No classification result was produced. Check --epochs; it may be 0.", flush=True)
        else:
            print("Best classification accuracy for %s: %.6f at epoch %s" % (obj_name, max_acc, best_epoch), flush=True)
            append_result_csv(
                args.result_csv,
                {
                    "dataset": args.dataset_name,
                    "obj": obj_name,
                    "task": "classification",
                    "split": args.test_split,
                    "best_epoch": best_epoch,
                    "selection_metric": "acc",
                    "acc": max_acc,
                    "auc_image": "",
                    "ap_image": "",
                    "f1_image": "",
                    "auc_pixel": "",
                    "ap_pixel": "",
                    "f1_pixel": "",
                    "pro_pixel": "",
                    "checkpoint": best_checkpoint,
                },
            )

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_id',  type=int, default=None)
    parser.add_argument('--sample_name', type=str, default='all')
    parser.add_argument('--mvtec_path', type=str,required=True)
    parser.add_argument('--generated_data_path', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, default=30)
    parser.add_argument('--log_interval', action='store', type=int, default=50)
    parser.add_argument('--image_size', action='store', type=int, default=512)
    parser.add_argument('--test_split', choices=['all', 'last_two_thirds'], default='all')
    parser.add_argument('--train_repeat', action='store', type=int, default=3)
    parser.add_argument('--seed', action='store', type=int, default=2026)
    parser.add_argument(
        "--reverse",
        action="store_true", default=False,
    )
    parser.add_argument('--checkpoint_path', default='checkpoints/classification', type=str)
    parser.add_argument('--result_csv', default='', type=str)
    parser.add_argument('--dataset_name', default='', type=str)

    args = parser.parse_args()
    seed_everything(args.seed)

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
        obj_batch = reversed(obj_batch)
    if args.sample_name!='all':
        obj_list=[args.sample_name]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    train_on_device(picked_classes, args)
#python train-classification.py
