import os

import pandas as pd

import torch
from torch.utils.data import DataLoader

import timm
from adamp import AdamP, SGDP

from dataset import *
from evaluation import *

@torch.no_grad()
def inference(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)  ##
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)

            age_outs = age_model(images)
            gender_outs = gender_model(images)
            mask_outs = mask_model(images)

            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            pred = age_preds + gender_preds * 3 + mask_preds * 6
            preds.extend(pred.cpu().numpy())
    
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'all_effi_b3_sgd_same.csv'), index=False)
    print(f'Inference Done!')

@torch.no_grad()
def inference_all(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'all_effi_b0_sgd'
    model_dir = os.path.join(model_dir, model_name)
    model = load_model(model_dir, num_classes, device).to(device)
    print(f'Using {model_name}..')
    model.eval()

    batch_size = 32

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.float().to(device)

            outs = model(images)
            pred = torch.argmax(outs, dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name}_same'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}')
    print(f'Inference Done!')

@torch.no_grad()
def inference_tta(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'all_effi_b3_sgd10'
    model_dir = os.path.join(model_dir, model_name)
    model = load_model(model_dir, num_classes, device).to(device)
    print(f'Using {model_name}..')
    model.eval()

    batch_size = 32

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDatasetForTTA(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            img1, img2, img3 = images
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)

            outs1 = model(img1)
            outs2 = model(img2)
            outs3 = model(img3)
            outs = outs1 + outs2 + outs3

            pred = torch.argmax(outs, dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name}_tta'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}.csv')
    print(f'Inference Done!')

@torch.no_grad()
def inference_multi(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'effi_b3_adamp_multi2'
    model_dir = os.path.join(model_dir, model_name)
    model = load_multi_model(model_dir, num_classes, device).to(device)
    print(f'Using {model_name}..')
    model.eval()

    batch_size = 32

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.float().to(device)

            outs = model(images)
            age_outs, gender_outs, mask_outs = outs

            age_preds = torch.argmax(age_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)

            pred = age_preds + gender_preds * 3 + mask_preds * 6
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name}'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}.csv')
    print(f'Inference Done!')

@torch.no_grad()
def inference_tta_with_multi(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model_name = 'effi_b3_adamp_multi3'
    model_dir = os.path.join(model_dir, model_name)
    model = load_multi_model(model_dir, num_classes, device).to(device)
    print(f'Using {model_name}..')
    model.eval()

    batch_size = 32

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDatasetForTTA(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            img1, img2, img3 = images
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)

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

            pred = age_preds + gender_preds * 3 + mask_preds * 6
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name}_tta'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}.csv')
    print(f'Inference Done!')

@torch.no_grad()
def inference_ensemble(data_dir, model_dir, output_dir):
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

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.float().to(device)

            outs_1 = model_1(images)
            outs_2 = model_2(images)

            outs = outs_1 + outs_2

            pred = torch.argmax(outs, dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name_1}_and_{model_name_2}_ensemble'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}')
    print(f'Inference Done!')

@torch.no_grad()
def inference_ensemble_tta(data_dir, model_dir, output_dir):
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

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDatasetForTTA(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            img1, img2, img3 = images
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)

            outs1_from1 = model_1(img1)
            outs2_from1 = model_1(img2)
            outs3_from1 = model_1(img3)
            outs_from1 = outs1_from1 + outs2_from1 + outs3_from1

            outs1_from2 = model_2(img1)
            outs2_from2 = model_2(img2)
            outs3_from2 = model_2(img3)
            outs_from2 = outs1_from2 + outs2_from2 + outs3_from2

            outs = outs_from1 + outs_from2

            pred = torch.argmax(outs, dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name_1}_and_{model_name_2}_ensemble_tta'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}')
    print(f'Inference Done!')

@torch.no_grad()
def inference_ensemble_with_multi_and_tta(data_dir, model_dir, output_dir):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    model_name_1 = 'all_effi_b3_sgd10' # all_effi_b3_sgd4
    model_dir_1 = os.path.join(model_dir, model_name_1)
    model_1 = load_model(model_dir_1, num_classes, device).to(device)

    model_name_2 = 'effi_b3_sgd_multi2'
    model_dir_2 = os.path.join(model_dir, model_name_2)
    model_2 = load_multi_model(model_dir_2, num_classes, device).to(device)

    model_name_3 = 'effi_b3_adamp_multi2'
    model_dir_3 = os.path.join(model_dir, model_name_3)
    model_3 = load_multi_model(model_dir_3, num_classes, device).to(device)

    print(f'Using {model_name_1} and {model_name_2} and {model_name_3}..')
    model_1.eval()
    model_2.eval()
    model_3.eval()

    batch_size = 16

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDatasetForTTA(img_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            img1, img2, img3 = images
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            img3 = img3.float().to(device)

            # model_1 : single
            out_1_from_1 = model_1(img1)  # [0.2, 0.2, .., 0.2,]
            out_2_from_1 = model_1(img2)
            out_3_from_1 = model_1(img3)
            outs_from_1 = out_1_from_1 + out_2_from_1 + out_3_from_1

            # model_2 : multi
            out1_from_2 = model_2(img1)
            age_outs1, gender_outs1, mask_outs1 = out1_from_2  # [0.2, 0.8], 
            out2_from_2 = model_2(img2)
            age_outs2, gender_outs2, mask_outs2 = out2_from_2
            out3_from_2 = model_2(img3)
            age_outs3, gender_outs3, mask_outs3 = out3_from_2

            age_outs = age_outs1 + age_outs2 + age_outs3
            gender_outs = gender_outs1 + gender_outs2 + gender_outs3
            mask_outs = mask_outs1 + mask_outs2 + mask_outs3

            outs_from_2 = [[0]*18 for _ in range(len(img1))]
            outs_from_2 = torch.FloatTensor(outs_from_2).to(device)
            for i, (ages, genders, masks) in enumerate(zip(age_outs, gender_outs, mask_outs)):
                temp = [0]*18
                temp = torch.FloatTensor(temp).to(device)
                for age_idx, age_out in enumerate(ages):
                    for gender_idx, gender_out in enumerate(genders):
                        for mask_idx, mask_out in enumerate(masks):
                            idx = age_idx + gender_idx * 3 + mask_idx * 6
                            temp[idx] = age_out * gender_out * mask_out
                outs_from_2[i] = temp
            
            # model_3 : multi and no TTA
            outs_from_3 = model_3(img1)
            age_outs, gender_outs, mask_outs = outs_from_3

            outs_from_3 = [[0]*18 for _ in range(len(img1))]
            outs_from_3 = torch.FloatTensor(outs_from_3).to(device)
            for i, (ages, genders, masks) in enumerate(zip(age_outs, gender_outs, mask_outs)):
                temp = [0]*18
                temp = torch.FloatTensor(temp).to(device)
                for age_idx, age_out in enumerate(ages):
                    for gender_idx, gender_out in enumerate(genders):
                        for mask_idx, mask_out in enumerate(masks):
                            idx = age_idx + gender_idx * 3 + mask_idx * 6
                            temp[idx] = age_out * gender_out * mask_out
                outs_from_3[i] = temp

            outs_from_2 = outs_from_2 / 1000
            outs_from_3 = outs_from_3 / 1000
            outs = outs_from_1 + outs_from_2 + outs_from_3

            pred = torch.argmax(outs, dim=-1)
            pred = pred.long()
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    file_name = f'{model_name_1}_and_{model_name_2}_and_{model_name_3}_ensemble_tta'
    info.to_csv(os.path.join(output_dir, f'{file_name}.csv'), index=False)
    print(f'Save at {file_name}')
    print(f'Inference Done!')

if __name__ == '__main__':
    data_dir = '/opt/ml/input/data/eval'
    model_dir = '/opt/ml/model'
    output_dir = '/opt/ml/output'

    os.makedirs(output_dir, exist_ok=True)

    # print('inference..')
    # inference_all(data_dir, model_dir, output_dir)

    # print('inference with tta..')
    # inference_tta(data_dir, model_dir, output_dir)

    print('inference with tta and multi..')
    inference_tta_with_multi(data_dir, model_dir, output_dir)

    # print('inference with multi..')
    # inference_multi(data_dir, model_dir, output_dir)

    # print('inference with ensemble..')
    # inference_ensemble(data_dir, model_dir, output_dir)

    # print('inference with ensemble and tta')
    # inference_ensemble_tta(data_dir, model_dir, output_dir)

    # print('inference with ensemble, multi and tta..')
    # inference_ensemble_with_multi_and_tta(data_dir, model_dir, output_dir)