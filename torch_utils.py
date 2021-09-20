import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained)
        n_features = backbone.fc.in_features
        self.model = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, CFG.target_size)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward_features(self, x):
        x = self.model(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats
class CFG:
    debug=False
    num_workers=0
    model_name='resnext50_32x4d'
    size=450
    batch_size=1
    seed=42
    target_size=5
    target_col='label'
    train=False
    pretrained=False
    inference=True

class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                #y_preds = model(images)
                y_preds,_ = model(images) #snapmix
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    # pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for imgs in data_loader:
        imgs = imgs.to(device).float()
        
        #image_preds = model(imgs)
        image_preds, _ = model(imgs)   #for snapmix inference
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
disease_dict = {
    0:"Cassava Bacterial Blight (CBB)",
    1:"Cassava Brown Streak Disease (CBSD)",
    2:"Cassava Green Mottle (CGM)",
    3:"Cassava Mosaic Disease (CMD)",
    4:"Healthy"
}


def get_prediction():
    seed_everything(CFG.seed)
    device = torch.device('cpu')
    model = CustomResNext().to(device)
    model1 = torch.load('resnext50_32x4d_fold0_epoch9_best.pth',map_location ='cpu')
    model1 = model1['model']
    model.load_state_dict(model1)
    # print(model.keys())
    # return
    # model.load_state_dict(torch.load('resnext50_32x4d_fold0_epoch9_best.pth')['model'])
    test = pd.DataFrame()
    test['image_id'] = list(os.listdir('./testimages/'))    
    test_ds = CassavaDataset(test, './testimages/', transforms=get_inference_transforms(), output_label=False)
    tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
            pin_memory=False,
        )
    tst_preds = []
    with torch.no_grad():
        # tst_preds += inference_one_epoch(model, tst_loader, device)
        tst_preds = inference_one_epoch(model, tst_loader, device)[0]
    del model
    torch.cuda.empty_cache()
    return disease_dict[np.argmax(tst_preds)]

# import io
# import torch 
# import torch.nn as nn 
# import torchvision.transforms as transforms 
# from PIL import Image




# # load model

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)  
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out

# input_size = 784 # 28x28
# hidden_size = 500 
# num_classes = 10
# model = NeuralNet(input_size, hidden_size, num_classes)

# PATH = "./mnist_ffn.pth"
# model.load_state_dict(torch.load(PATH))
# model.eval()

# # image -> tensor
# def transform_image(image_bytes):
#     transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                     transforms.Resize((28,28)),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.1307,),(0.3081,))])

#     image = Image.open(io.BytesIO(image_bytes))
#     return transform(image).unsqueeze(0)

# # predict
# def get_prediction(image_tensor):
#     images = image_tensor.reshape(-1, 28*28)
#     outputs = model(images)
#         # max returns (value ,index)
#     _, predicted = torch.max(outputs.data, 1)
#     return predicted