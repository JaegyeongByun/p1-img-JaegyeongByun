import os

from train import increment_path
from dataset import *

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import albumentations as A
import timm
from adamp import AdamP, SGDP

from torchvision import models
from torchvision import transforms
from torchvision.transforms import *
from torch.utils.tensorboard import SummaryWriter

from model import *

from collections import defaultdict
import pickle


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = TrainDatasetForThreeModel.decode_multi_class(gt)
        pred_decoded_labels = TrainDatasetForThreeModel.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def load_model(saved_model, num_classes, device):
    model = timm.create_model('efficientnet_b3', pretrained=True)
    model.classifier = nn.Sequential(
    nn.Linear(1536, 1000),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(1000, 500),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(500, num_classes),
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_multi_model(saved_model, num_classes, device):
    model = CustomResNext('efficientnet_b3', pretrained=True)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def evaluation(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    save_dir = increment_path(os.path.join(model_dir, '0405_val_log'))

    age_num_classes = 3
    age_model_dir = os.path.join(model_dir, 'age_log_v')
    age_model = load_model(age_model_dir, age_num_classes, device).to(device)

    gender_num_classes = 2
    gender_model_dir = os.path.join(model_dir, 'gender_log_v')
    gender_model = load_model(gender_model_dir, gender_num_classes, device).to(device)

    mask_num_classes = 3
    mask_model_dir = os.path.join(model_dir, 'mask_log_v')
    mask_model = load_model(mask_model_dir, mask_num_classes, device).to(device)

    age_model.eval()
    gender_model.eval()
    mask_model.eval()

    batch_size = 16
    # transform = transforms.Compose([
    #     ToTensor(),
    # ])

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.5),
#        A.CenterCrop(height=512, width=384, p=0.5),

    ])

    dataset = TrainDatasetForThreeModel(data_dir, category='all')
    dataset.set_transform(transform)
    
    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    logger = SummaryWriter(log_dir=save_dir)
    print_every = 100
    with torch.no_grad():
        val_acc_items = []
        figure = None
        for idx, val_batch in enumerate(val_loader):
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            age_outs = age_model(inputs)
            gender_outs = gender_model(inputs)
            mask_outs = mask_model(inputs)

            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            preds = age_preds + gender_preds * 3 + mask_preds * 6

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)
            
            val_acc = np.sum(val_acc_items) / len(val_set)
            if idx % print_every == 0:
                # print(f"labels: {labels}")
                # print(f"preds : {preds}")
                # print(
                #     f"{idx + 1}/{len(val_loader)}||"
                #     f"[Val] acc : {val_acc:4.2%}"
                # )

                inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                figure = grid_image(inputs_np, labels, preds, shuffle=False)
                logger.add_scalar("Val/accuracy", val_acc)
                logger.add_figure("results", figure)
                logger.add_figure("results", figure)

        val_acc = np.sum(val_acc_items) / len(val_set)
        print(
            f"[Val] acc : {val_acc:4.2%}"
        )
        print()

@torch.no_grad()
def evaluation_all(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'all_effi_b0_sgd'
    model_dir = os.path.join(model_dir, model_name)
    model = load_model(model_dir, num_classes, device).to(device)
    print(f'Using {model_name}..')

    model.eval()

    batch_size = 16
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.5),
#        A.CenterCrop(height=512, width=384, p=0.5),
    ])

    dataset = TrainDatasetForThreeModel(data_dir, category='all')
    dataset.set_transform(transform)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            inputs, labels = val_batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )    
    
    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name}'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_age_gender(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    save_dir = increment_path(os.path.join(model_dir, '0405_val_log'))

    age_gender_num_classes = 6
    age_gender_model_dir = os.path.join(model_dir, 'agegender_AgeGenderModel')
    age_gender_model = load_model(age_gender_model_dir, age_gender_num_classes, device).to(device)

    mask_num_classes = 3
    mask_model_dir = os.path.join(model_dir, 'mask_log_v')
    mask_model = load_model(mask_model_dir, mask_num_classes, device).to(device)

    age_gender_model.eval()
    mask_model.eval()

    batch_size = 16
    # transform = transforms.Compose([
    #     ToTensor(),
    # ])

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.5),
#        A.CenterCrop(height=512, width=384, p=0.5),

    ])

    dataset = TrainDatasetForThreeModel(data_dir, category='all')
    dataset.set_transform(transform)
    
    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    logger = SummaryWriter(log_dir=save_dir)
    print_every = 100
    with torch.no_grad():
        val_acc_items = []
        figure = None
        for idx, val_batch in enumerate(val_loader):
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            age_gender_outs = age_gender_model(inputs)
            mask_outs = mask_model(inputs)

            age_gender_preds = torch.argmax(age_gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            preds = age_gender_preds + mask_preds * 6

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)
            
            val_acc = np.sum(val_acc_items) / len(val_set)
            if idx % print_every == 0:
                # print(f"labels: {labels}")
                # print(f"preds : {preds}")
                # print(
                #     f"{idx + 1}/{len(val_loader)}||"
                #     f"[Val] acc : {val_acc:4.2%}"
                # )

                inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                #inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                figure = grid_image(inputs_np, labels, preds, shuffle=False)
                logger.add_scalar("Val/accuracy", val_acc)
                logger.add_figure("results", figure)
                logger.add_figure("results", figure)

        val_acc = np.sum(val_acc_items) / len(val_set)
        print(
            f"[Val] acc : {val_acc:4.2%}"
        )
        print()

@torch.no_grad()
def evaluation_multi(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'effi_b3_sgd_multi2'
    model_dir = os.path.join(model_dir, model_name)
    print(f'Using {model_name}..')
    model = load_multi_model(model_dir, num_classes, device).to(device)
    model.eval()

    batch_size = 16

    transform = A.Compose([
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
    ])

    dataset = TrainDatasetForMulti(data_dir, val_ratio=0.2, tta=False)
    dataset.set_transform(transform)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            imgs, labels = val_batch
            imgs = imgs.float().to(device)

            age_labels, gender_labels, mask_labels = labels
            age_labels = age_labels.long().to(device)
            gender_labels = gender_labels.long().to(device)
            mask_labels = mask_labels.long().to(device)
            
            outs = model(imgs)
            age_outs, gender_outs, mask_outs = outs

            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            preds = age_preds + gender_preds * 3 + mask_preds * 6
            labels = age_labels + gender_labels * 3 + mask_labels * 6

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1
            
    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )

    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name}_tta'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_tta_with_multi(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'effi_b3_adamp_multi2'
    model_dir = os.path.join(model_dir, model_name)
    print(f'Using {model_name}..')
    model = load_multi_model(model_dir, num_classes, device).to(device)
    model.eval()

    batch_size = 16

    dataset = TrainDatasetForMulti(data_dir, val_ratio=0.2, tta=True)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            imgs, labels = val_batch

            img1, img2, img3 = imgs
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)

            age_labels, gender_labels, mask_labels = labels
            age_labels = age_labels.long().to(device)
            gender_labels = gender_labels.long().to(device)
            mask_labels = mask_labels.long().to(device)
            
            out1 = model(img1)
            age_outs1, gender_outs1, mask_outs1 = out1
            out2 = model(img2)
            age_outs2, gender_outs2, mask_outs2 = out2
            out3 = model(img3)
            age_outs3, gender_outs3, mask_outs3 = out3

            age_outs = age_outs1 + age_outs2 + age_outs3
            gender_outs = gender_outs1 + gender_outs2 + gender_outs3
            mask_outs = mask_outs1 + mask_outs2 + mask_outs3

            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            preds = age_preds + gender_preds * 3 + mask_preds * 6
            labels = age_labels + gender_labels * 3 + mask_labels * 6

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1
            

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )

    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name}_tta'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_tta(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'all_effi_b3_sgd10'
    model_dir = os.path.join(model_dir, model_name)
    print(f'Using {model_name}..')
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    batch_size = 16

    dataset = TrainDatasetForThreeModel(data_dir, category='all', tta=True)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            img1, img2, img3, labels = val_batch
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)
            labels = labels.long().to(device)
            
            out1 = model(img1)
            out2 = model(img2)
            out3 = model(img3)
            outs = out1 + out2 + out3
            preds = torch.argmax(outs, dim=-1)
            
            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1
            

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )

    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name}_tta'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_ensemble(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name_1 = 'all_effi_b3_sgd4'
    model_dir_1 = os.path.join(model_dir, model_name_1)
    model_1 = load_model(model_dir_1, num_classes, device).to(device)

    model_name_2 = 'all_effi_b3_sgd10'
    model_dir_2 = os.path.join(model_dir, model_name_2)
    model_2 = load_model(model_dir_2, num_classes, device).to(device)

    print(f'Using {model_name_1} and {model_name_2}..')
    model_1.eval()
    model_2.eval()

    batch_size = 16
    transform = A.Compose([
        A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246))
    ])

    dataset = TrainDatasetForThreeModel(data_dir, category='all')
    dataset.set_transform(transform)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            inputs, labels = val_batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            outs_from1 = model_1(inputs)
            outs_from2 = model_2(inputs)
            outs = outs_from1 + outs_from2

            preds = torch.argmax(outs, dim=-1)

            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )    
    
    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name_1}_{model_name_2}'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_ensemble_tta(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name_1 = 'all_effi_b3_sgd4'
    model_dir_1 = os.path.join(model_dir, model_name_1)
    model_1 = load_model(model_dir_1, num_classes, device).to(device)

    model_name_2 = 'all_effi_b3_sgd10'
    model_dir_2 = os.path.join(model_dir, model_name_2)
    model_2 = load_model(model_dir_2, num_classes, device).to(device)

    print(f'Using {model_name_1} and {model_name_2}..')
    model_1.eval()
    model_2.eval()

    batch_size = 16

    dataset = TrainDatasetForThreeModel(data_dir, category='all', tta=True)

    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            img1, img2, img3, labels = val_batch
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)
            labels = labels.long().to(device)
            
            out1_from1 = model_1(img1)
            out2_from1 = model_1(img2)
            out3_from1 = model_1(img3)
            outs_from1 = out1_from1 + out2_from1 + out3_from1

            out1_from2 = model_2(img1)
            out2_from2 = model_2(img2)
            out3_from2 = model_2(img3)
            outs_from2 = out1_from2 + out2_from2 + out3_from2
 
            outs = outs_from1 + outs_from2

            preds = torch.argmax(outs, dim=-1)
            
            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1
            

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )

    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name_1}_{model_name_2}_tta'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

@torch.no_grad()
def evaluation_ensemble_multi_tta(data_dir, model_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name_1 = 'all_effi_b3_sgd10'
    model_dir_1 = os.path.join(model_dir, model_name_1)
    model_1 = load_model(model_dir_1, num_classes, device).to(device)

    model_name_2 = 'effi_b3_adamp_multi2'
    model_dir_2 = os.path.join(model_dir, model_name_2)
    model_2 = load_multi_model(model_dir_2, num_classes, device).to(device)

    model_name_3 = 'effi_b3_sgd_multi4'
    model_dir_3 = os.path.join(model_dir, model_name_3)
    model_3 = load_multi_model(model_dir_3, num_classes, device).to(device)


    print(f'Using {model_name_1} and {model_name_2} and {model_name_3}..')
    model_1.eval()
    model_2.eval()
    model_3.eval()

    batch_size = 16

    dataset = TrainDatasetForMulti(data_dir, tta=True)
    _, val_set = dataset.split_dataset()
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating evaluation results..")
    val_acc_items = []
    figure = None

    all_answer = defaultdict(int)
    wrong_answer = defaultdict(int)
    label_pred = defaultdict(list)
    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            (img1, img2, img3), (age_labels, gender_labels, mask_labels) = val_batch
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)
            age_labels = age_labels.long().to(device)
            gender_labels = gender_labels.long().to(device)
            mask_labels = mask_labels.long().to(device)
            
            # model_1 : single
            out1_from1 = model_1(img1)
            out2_from1 = model_1(img2)
            out3_from1 = model_1(img3)
            outs_from1 = out1_from1 + out2_from1 + out3_from1

            # model_2 : multi
            out1_from2 = model_2(img1)
            age_outs1, gender_outs1, mask_outs1 = out1_from2  # [0.2, 0.8], 
            out2_from2 = model_2(img2)
            age_outs2, gender_outs2, mask_outs2 = out2_from2
            out3_from2 = model_2(img3)
            age_outs3, gender_outs3, mask_outs3 = out3_from2

            age_outs = age_outs1 + age_outs2 + age_outs3
            gender_outs = gender_outs1 + gender_outs2 + gender_outs3
            mask_outs = mask_outs1 + mask_outs2 + mask_outs3

            outs_from2 = [[0]*18 for _ in range(len(img1))]
            outs_from2 = torch.FloatTensor(outs_from2).to(device)
            for i, (ages, genders, masks) in enumerate(zip(age_outs, gender_outs, mask_outs)):
                temp = [0]*18
                temp = torch.FloatTensor(temp).to(device)
                for age_idx, age_out in enumerate(ages):
                    for gender_idx, gender_out in enumerate(genders):
                        for mask_idx, mask_out in enumerate(masks):
                            idx = age_idx + gender_idx * 3 + mask_idx * 6
                            temp[idx] = age_out * gender_out * mask_out
                outs_from2[i] = temp

            # model_3 : multi no tta
            outs_from3 = model_3(img1)
            age_outs, gender_outs, mask_outs = outs_from3  # [0.2, 0.8], 

            outs_from3 = [[0]*18 for _ in range(len(img1))]
            outs_from3 = torch.FloatTensor(outs_from3).to(device)
            for i, (ages, genders, masks) in enumerate(zip(age_outs, gender_outs, mask_outs)):
                temp = [0]*18
                temp = torch.FloatTensor(temp).to(device)
                for age_idx, age_out in enumerate(ages):
                    for gender_idx, gender_out in enumerate(genders):
                        for mask_idx, mask_out in enumerate(masks):
                            idx = age_idx + gender_idx * 3 + mask_idx * 6
                            temp[idx] = age_out * gender_out * mask_out
                outs_from3[i] = temp
            
            
            outs_from2 = outs_from2 / 1000
            outs_from3 = outs_from3 / 1000

            outs = outs_from1 + outs_from2 + outs_from3

            preds = torch.argmax(outs, dim=-1)
            preds = preds.long()
            labels = age_labels + gender_labels * 3 + mask_labels * 6
            acc_item = (labels == preds).sum().item()
            val_acc_items.append(acc_item)

            for pred, label in zip(preds, labels):
                pred = pred.cpu().numpy()
                pred = int(pred)
                label = label.cpu().numpy()
                label = int(label)
                if pred != label:
                    wrong_answer[pred] += 1
                    label_pred[label].append(pred)
                all_answer[pred] += 1
            

    val_acc = np.sum(val_acc_items) / len(val_set)
    print(
        f"[Val] acc : {val_acc:4.2%}"
    )

    eval_dir = '/opt/ml/eval'
    file_name = f'{model_name_1}_{model_name_2}_{model_name_3}_tta'
    with open(os.path.join(eval_dir, f'{file_name}_all.pickle'), 'wb') as fw:
        pickle.dump(all_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_wrong.pickle'), 'wb') as fw:
        pickle.dump(wrong_answer, fw)
    with open(os.path.join(eval_dir, f'{file_name}_label_pred.pickle'),'wb') as fw:
        pickle.dump(label_pred, fw)

    print(f'Save at {file_name}..')
    print()

if __name__ == '__main__':
    data_dir = '/opt/ml/input/data/train/images'
    model_dir = '/opt/ml/model'

    # print('Using Three model..')
    # evaluation(data_dir, model_dir)

    # print('Using One model..')
    # evaluation_all(data_dir, model_dir)

    # print('Using TTA..')
    # evaluation_tta(data_dir, model_dir)

    # print('Using multi..')
    # evaluation_multi(data_dir, model_dir)

    # print('Using ensemble')
    # evaluation_ensemble(data_dir, model_dir)

    # print('Using ensemble and tta')
    # evaluation_ensemble_tta(data_dir, model_dir)

    print('Using TTA with multi..')
    evaluation_tta_with_multi(data_dir, model_dir)

    print('Using ensemble, tta, and multi')
    evaluation_ensemble_multi_tta(data_dir, model_dir)