import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
import sys
import pdb
import time
import random

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn import functional as F
import torchvision.models as models

import albumentations
from albumentations.pytorch import ToTensor

''' 1. import the EfficientNet as CLS model '''
pack_path = 'E:/datasets/steel-defect-detection/packages/EfficientNet-PyTorch-master'
sys.path.append(pack_path)

from efficientnet_pytorch import EfficientNet

''' 2. Dataset setup '''

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(1998)

train_transform = albumentations.Compose([
        albumentations.Resize(128, 800),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.OneOf([
                                albumentations.RandomBrightnessContrast(),
                                albumentations.RandomGamma(),
                             ], p=0.3),
        albumentations.OneOf([
                                albumentations.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                albumentations.GridDistortion(),
                                albumentations.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                             ], p=0.3),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
    ])

test_transform = albumentations.Compose([
        albumentations.Resize(128, 800),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
    ])


class SteelDataset(Dataset):
    def __init__(self, df, df2, model):
        
        self.model = model
        if self.model == 'Train':
            #self.uid = list(np.concatenate([np.load('E:/datasets/steel-defect-detection/packages/split-data/train_data_1.npy' , allow_pickle=True)]))
            self.uid = list(np.concatenate([np.load('E:/datasets/steel-defect-detection/packages/csv_plus_14/train_data_3_new.npy' , allow_pickle=True)]))
            self.transform = train_transform
        elif self.model == 'Valid':
            #self.uid = list(np.concatenate([np.load('E:/datasets/steel-defect-detection/packages/split-data/valid_data_1.npy' , allow_pickle=True)]))
            self.uid = list(np.concatenate([np.load('E:/datasets/steel-defect-detection/packages/fold-5/valid_data_3.npy' , allow_pickle=True)]))
            self.transform = test_transform

        df.fillna('', inplace=True)
        self.df = df
        self.df2 = df2

    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        image_id = self.uid[index]
        if self.model == 'Train':

            if image_id in self.df2['ImageId'].values:

                classes = self.df2.loc[self.df2['ImageId']==image_id, 'ClassId'].values[0]
                label = [1,0,0,0] if classes == 1 else [0,0,0,1]
                image = cv2.imread(f'E:/datasets/steel-defect-detection/data/test_images/{image_id}')
            else:
                rle = [
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
                ]

                label = [1 if r != '' else 0 for r in rle]
                image = cv2.imread(f'E:/datasets/steel-defect-detection/data/train_images/{image_id}')
        else:
            rle = [
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
                    self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
                ]

            label = [1 if r != '' else 0 for r in rle]
            image = cv2.imread(f'E:/datasets/steel-defect-detection/data/train_images/{image_id}')

        label = np.array(label)
        label = torch.from_numpy(label).float()

        # image = cv2.imread(f'E:/datasets/steel-defect-detection/data/train_images/{image_id}')
        augment = self.transform(image=image)
        image = augment['image'].transpose(2, 0, 1)
        image = torch.from_numpy(image)

        return image_id, image, label

''' 3. Dataloader '''

df = pd.read_csv('E:/datasets/steel-defect-detection/data/train.csv')
df14 = pd.read_csv('E:/datasets/steel-defect-detection/packages/csv_plus_14/df14.csv')

train_dataset = SteelDataset(df, df14, 'Train')
valid_dataset = SteelDataset(df, df14, 'Valid')

train_loader  = DataLoader(
        train_dataset,
        batch_size  = 24,
        shuffle = False,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
    )

valid_loader = DataLoader(
        valid_dataset,
        shuffle = False,
        batch_size  = 24,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
    )


''' 4. Training process and Valid process '''
# check_point = torch.load('E:/datasets/steel-defect-detection/pth/B3_256_1600_fold-2_epoch29.pth')
check_point = torch.load('E:/datasets/steel-defect-detection/pth/efficientnet-b3-5fb5a3c3.pth')

cls_model = EfficientNet.from_name('efficientnet-b3')

# for param in cls_model.parameters():
#     param.requires_grad = False
cls_model.load_state_dict(check_point)
in_features = cls_model._fc.in_features
cls_model._fc = nn.Linear(in_features, 4)
# cls_model.load_state_dict(check_point['state_dict'])

if torch.cuda.device_count() > 3:
    cls_model = nn.DataParallel(cls_model)
    print('4 GPUs has been enable!')
else:
    print('Multi GPUs error!')

if torch.cuda.is_available():
    cls_model.cuda()
    print(f'The data has been put on CUDA!')
else:
    print(f'Data to CUDA error!')


criterion = nn.BCEWithLogitsLoss()

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

def valid_running():
    t_label = np.ones(4).reshape(1,4)*9
    p_label = np.ones(4).reshape(1,4)*9
    
    cls_model.eval()
    valid_loss = 0.
    
    for t, (image_id, image, truth_label) in enumerate(valid_loader):
        image = image.cuda()
        truth_label = truth_label.cuda()
        
        with torch.no_grad():
            logit = cls_model(image)
            loss = criterion(logit, truth_label)
            
            pred = torch.sigmoid(logit)
            truth_label = truth_label.cpu().numpy()
            preds_label = pred.cpu().numpy()
            
            t_label = np.vstack((t_label, truth_label))
            p_label = np.vstack((p_label, preds_label))
            
        valid_loss += loss.item() / len(valid_loader)
        
    p_label = p_label[1:]
    t_label = t_label[1:]
    p_label = (p_label > 0.5)
    
    TP = np.sum(np.multiply(t_label, p_label))    # pred=1 & truth=1
    TN = np.sum(np.logical_and(np.equal(t_label, 0), np.equal(p_label, 0)))    # pred=0 & truth=0
    FP = np.sum(np.logical_and(np.equal(t_label, 0), np.equal(p_label, 1)))    # pred=1 & truth=0
    FN = np.sum(np.logical_and(np.equal(t_label, 1), np.equal(p_label, 0)))    # pred=0 & truth=1
    
    acc = (TP+TN) / (TP+FP+FN+TN)
    acc2 = (t_label==p_label).sum(axis=1)
    acc2 = np.sum(acc2==4)
    acc2 = acc2 / len(t_label)
    
    P=TP / (TP+FP)
    R=TP / (TP+FN)
    
    F1=2*P*R/(P+R)
    
    num_Pos = np.sum(t_label==1)
    num_Neg = np.sum(t_label==0)
        
    return valid_loss, acc2, F1, TP, TN, num_Pos, num_Neg

def train_running(epoch, learning_rate):
    
    optimizer = torch.optim.SGD(cls_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    cls_model.train()
    train_loss = 0.
    optimizer.zero_grad()
    
    for t, (image_id, image, truth_label) in enumerate(train_loader):
        image = image.cuda()
        truth_label = truth_label.cuda()
        
        logit = cls_model(image)
        loss = criterion(logit, truth_label)
            
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item() / len(train_loader)
        
    return train_loss

''' 5. Training '''
val_best_loss = 100.
best_acc2 = 0.1
num_epoch = 42
learning_rate = 3e-3
print('The training process has been running')

for epoch in range(1, num_epoch+1):
    start_time2 = time.time()

    if epoch % 3 == 0:
        learning_rate *= 0.4
    if epoch % 27 == 0:
        learning_rate = 3e-4

    train_loss = train_running(epoch, learning_rate)
    valid_loss, acc2, F1, TP, TN, num_P, num_N = valid_running()
    
    epoch_time = time.time() - start_time2

    print(f'epoch={epoch}/{num_epoch},|t_loss={train_loss:.5f},|v_loss={valid_loss:.5f},|acc2={acc2:.5f},|F1={F1:.5f},|TP={TP},|TN={TN},|num_P={num_P},|num_N={num_N},|time={epoch_time:.1f},|lr={learning_rate:.8f}')
    if valid_loss <= val_best_loss or acc2 >= best_acc2:
        if valid_loss <= val_best_loss:
            val_best_loss = valid_loss
            print(f'The model has been saved by valid-loss={val_best_loss}')
        if acc2 >= best_acc2:
            best_acc2 = acc2
            print(f'The model has been saved by acc2')

        state = {
                  'state_dict': cls_model.module.state_dict(),
                  'val_loss': valid_loss
                }
        
        torch.save(state, f'E:/datasets/steel-defect-detection/pth/B3_128-800_plus14_epoch{epoch}.pth')
            
    # scheduler.step()
    
print(f'The training process has done')

