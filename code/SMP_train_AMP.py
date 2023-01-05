import os
import random
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

import numpy as np

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

#!pip install albumentations==0.4.6
import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb

from dataloader import CustomDataLoader
from SMP_dataset_AMP import train, validation, save_model
from torch.optim.lr_scheduler import *


print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 10   # Mini-batch size
num_epochs = 50
learning_rate = 0.0001

# seed 고정
random_seed = 21
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# train.json / validation.json / test.json 디렉토리 설정
dataset_path  = '/opt/ml/input/data'
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
test_path = dataset_path + '/test.json'

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


import albumentations as A
from albumentations.pytorch import ToTensorV2
# from copy_paste import CopyPaste

train_transform = A.Compose([
                            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.7),
                            A.ShiftScaleRotate(p=0.7),
                            A.Blur(p=0.5),
                            A.Perspective(p=0.5),
                            A.GridDistortion(p=0.5),
                            A.MaskDropout(max_objects=2, mask_fill_value=0, p=0.6),
                            A.RandomSizedCrop((512-256, 512), 512, 512, p=0.8),
                            A.Flip(p=0.5),
                            # CopyPaste(),
                            ToTensorV2(),
                            ])

val_transform = A.Compose([
                            ToTensorV2()
                          ])

# create own Dataset 1 (skip)
# validation set을 직접 나누고 싶은 경우
# random_split 사용하여 data set을 8:2 로 분할
# train_size = int(0.8*len(dataset))
# val_size = int(len(dataset)-train_size)
# dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# create own Dataset 2
# train dataset
train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)

# validation dataset
val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)


# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           drop_last=True,
                                           collate_fn=collate_fn)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn)


import torch.nn as nn
import segmentation_models_pytorch as smp
from models import build

build.register_encoder()

# model 불러오기
# 출력 label 수 정의 (classes=11)

model = smp.PAN(
			encoder_name="swin_encoder",
			encoder_weights="imagenet",
            encoder_output_stride=32,
			in_channels=3,
			classes=11
)

# 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
x = torch.randn([2, 3, 512, 512])
print(f"input shape : {x.shape}")
out = model(x)
print(f"output shape : {out.size()}")


val_every = 1

saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
    
    
# Loss function 정의
criterion = criterion = nn.CrossEntropyLoss()

# Optimizer 정의
optimizer = torch.optim.AdamW(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

# Scheduler 정의
scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

wandb.init(project="semantic_segmentation_baseline", entity="cv_09_semanticsegmentation", name="SwinL_AUGMANY_FCLoss_AMP_bat10")    ########## 바꿔주세요 ##########
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
    }
train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, val_every, device)