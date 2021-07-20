import os
import glob
import json
import re
import argparse
from pathlib import Path

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import timm
from adamp import AdamP, SGDP

from torchvision import transforms
from torchvision.transforms import *
from torchvision import models

from dataset import TrainDatasetForThreeModel, TrainDatasetForMulti
from model import *
from loss import *

from tqdm.notebook import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def grid_image(np_images, category, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        title = f"{category} - gt: {gt}, pred: {pred}"

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary, vmin=0, vmax=1)

    return figure

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def custom_loss(output, target):
    # mask_loss = nn.CrossEntropyLoss()(output[0], target[0])
    # gender_loss = nn.CrossEntropyLoss()(output[1], target[1])
    # age_loss = nn.CrossEntropyLoss()(output[2], target[2])

    mask_loss = LabelSmoothingLoss()(output[0], target[0])
    gender_loss = LabelSmoothingLoss()(output[1], target[1])
    age_loss = LabelSmoothingLoss()(output[2], target[2])

    return mask_loss + gender_loss + age_loss, mask_loss.item(), gender_loss.item(), age_loss.item()

def train_multi(data_dir, model_dir, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, 'effi_b3_adamp_multi'))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 16

    transform = A.Compose([
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
    ])

    dataset = TrainDatasetForMulti(data_dir, val_ratio=0.2, upsampling=False)
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    model = CustomResNext('efficientnet_b3', pretrained=True)
    model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optimizer = AdamP(model.parameters(), lr=0.00005, betas=(0.9, 0.999), weight_decay=1e-2)
    epochs = 3

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    log_interval = 100
    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, (age_labels, gender_labels, mask_labels) = train_batch
            inputs = inputs.float().to(device)
            age_labels = age_labels.long().to(device)
            gender_labels = gender_labels.long().to(device)
            mask_labels = mask_labels.long().to(device)
            
            optimizer.zero_grad()

            outs = model(inputs)
            loss = custom_loss(outs, (age_labels, gender_labels, mask_labels))[0]

            age_outs, gender_outs, mask_outs = outs
            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            preds = age_preds + gender_preds * 3 + mask_preds * 6
            labels = age_labels + gender_labels * 3 + mask_labels * 6
            matches += (preds == labels).sum().item()
            if idx % log_interval == 0:
                train_loss = loss_value / log_interval
                train_acc = matches / batch_size / log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (age_labels, gender_labels, mask_labels) = val_batch
                inputs = inputs.float().to(device)
                age_labels = age_labels.long().to(device)
                gender_labels = gender_labels.long().to(device)
                mask_labels = mask_labels.long().to(device)
                                
                outs = model(inputs)
                age_outs, gender_outs, mask_outs = outs
                age_preds = torch.argmax(age_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                mask_preds = torch.argmax(mask_outs, dim=-1)

                preds = age_preds + gender_preds * 3 + mask_preds * 6
                labels = age_labels + gender_labels * 3 + mask_labels * 6

                loss = custom_loss(outs, (age_labels, gender_labels, mask_labels))[0]
                loss_item = loss.item()

                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model in {save_dir}..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            print()

    print(f"saving model in {save_dir}")
    torch.save(model.state_dict(), f"{save_dir}/last.pth")

def train(data_dir, model_dir, category, args):
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, category+'_effi_b3_sgd'))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 16

    # transform = transforms.Compose([
    #     ToTensor(),
    #     Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))

    # ])

    # transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.ShiftScaleRotate(p=0.5),
    #     A.CenterCrop(height=384, width=384, p=1),
    #     A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
    # ])

    transform = A.Compose([
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
    ])

    dataset = TrainDatasetForThreeModel(data_dir, category=category, upsampling=False, val_ratio=0.2)
    dataset.set_transform(transform)

    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    num_classes = dataset.num_classes
    # model = models.googlenet(pretrained=True)
    # model.fc = nn.Sequential(    
    #     nn.Linear(1024, 1000),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(1000, 1000),
    #     nn.ReLU(True),
    #     nn.Dropout(),
    #     nn.Linear(1000, num_classes),
    # )

    if category == 'age_gender':
        model = AgeGenderModel(img_size=384, num_classes=6)
        epochs = 5
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    
    else:
        model = timm.create_model('efficientnet_b3', pretrained=True)
        n_features = 1536
        model.classifier = nn.Sequential(
        nn.Linear(n_features, 1000),
        nn.ReLU(True),
        nn.Dropout(p=0.6),
        nn.Linear(1000, 500),
        nn.ReLU(True),
        nn.Dropout(p=0.7),
        nn.Linear(500, num_classes),
        )
        epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    model.to(device)

    # weights = dataset.weights
    # weights = torch.Tensor(weights).to(device)

    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.2)
    # criterion = F1Loss(classes=num_classes, epsilon=1e-7)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = AdamP(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2)
    # optimizer = SGDP(params, lr=0.1, weight_decay=1e-5, momentum=0.9, nesterov=True)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    log_interval = 100
    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(tqdm(train_loader)):
            inputs, labels = train_batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if idx % log_interval == 0:
                train_loss = loss_value / log_interval
                train_acc = matches / batch_size / log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    #inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, category, labels, preds, n=4, shuffle=False)
                
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model in {save_dir}..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            logger.add_figure("results", figure, epoch)
            print()

    print(f"saving model in {save_dir}")
    torch.save(model.state_dict(), f"{save_dir}/last.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

    args = parser.parse_args()
    print(args)

    data_dir = '/opt/ml/input/data/train/images'
    model_dir = '/opt/ml/model'

    # print('Starting train AgeGenderModel...')
    # train(data_dir, model_dir, 'age_gender', args)

    # print('Starting train age model...')
    # train(data_dir, model_dir, 'age', args)

    # print('Starting train gender model...')
    # train(data_dir, model_dir, 'gender', args)

    # print('Starting train mask model...')
    # train(data_dir, model_dir, 'mask', args)

    # print('Starting train all model...')
    # train(data_dir, model_dir, 'all', args)

    print('Starting train multi model...')
    train_multi(data_dir, model_dir, args)
