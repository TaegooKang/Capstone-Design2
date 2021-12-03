import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd

from PIL import Image

class CXR(Dataset):

    def __init__(self, train=True, transform=None, class_type=None): # class type: 'normal' or 'abnormal'

        self.path_csv = '/root/ktg/Capstone2/dataset/csv'
        self.train = train
        if train:
            filename = 'train.csv'
        else:
            filename = 'test.csv'
        self.transform = transform

        info = pd.read_csv(os.path.join(self.path_csv, filename))
        info_0 = info[info['label']==0]
        info_1 = info[info['label']==1]
        info_1 = info_1.head(int(info_1.shape[0]/2))
        info = pd.concat([info_0, info_1])
        
        if class_type == 'normal':
            info = info[info['label']==0]
        elif class_type == 'abnormal':
            info = info[info['label']==1]
        
        self.img_list = list(info['filepath'])
        self.label_list = list(info['label'])
       
    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        label = torch.FloatTensor([self.label_list[index]])

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def undersampling(self):
        if self.train:
            info = pd.read_csv(os.path.join(self.path_csv, 'train.csv'))
            info_0 = info[info['label']==0]
            info_1 = info[info['label']==1]
            info_1 = info_1.head(int(info_1.shape[0]/2))
            info_0 = info_0.sample(frac=1).head(info_1.shape[0])
            info = pd.concat([info_0, info_1])

            self.img_list = list(info['filepath'])
            self.label_list = list(info['label'])


    def __len__(self):
        return len(self.img_list)
    
