import torch
from torch.utils.data import Dataset
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import sys
import os
# import matplotlib.pyplot as plt

def is_ordered(l):
    idx_list = [t[0] for t in l]
    if idx_list == sorted(idx_list) or idx_list == sorted(idx_list, reverse=True):
        return True
    else:
        return False


class tuple_dataset(Dataset):
    def __init__(
        self,
        data_path = "/home/zzhan226/code/dataset/",
        stage = "train",
        transforms = None,
    ): 
        super().__init__()
        self.stage = stage
        self.data = {'a':[],'b':[],'c':[]}
        
        self.transforms =transforms
        if stage == 'train':
            for img_name in os.listdir(data_path):
                if img_name == 'test': continue
                try:
                    img = Image.open(data_path + img_name)
                except OSError:
                    print("Cannot load : {}".format(data_path + img_name))
                
                if 'a' in img_name:
                    self.data['a'].append(img)
                elif 'b' in img_name:
                    self.data['b'].append(img)
                else:
                    self.data['c'].append(img)

        elif stage == 'test':
            for img_name in os.listdir(data_path + 'test/'):
                img = Image.open(data_path + 'test/' + img_name)
                if 'a' in img_name:
                    self.data['a'].append(img)
                elif 'b' in img_name:
                    self.data['b'].append(img)
                else:
                    self.data['c'].append(img)
            
        else:
            raise Exception('stage has to be train or test')
        
        # print(len(self.data['a']),len(self.data['b']),len(self.data['c']))s
        
    
    def __len__(self):
        return len(self.data['a'])
    
    def __getitem__(self, index: int):

        img_a, img_b, img_c = self.data['a'][index], self.data['b'][index],self.data['c'][index]
        img_tuple = [(1,img_a),(2,img_b),(3, img_c)]
        if np.random.random() > 0.5: #return positive tuple
            label = 1
            if np.random.random()>0.5:
                img_tuple.sort(key=lambda x:x[0],reverse=True)
        else:
            label = 0
            while is_ordered(img_tuple):
                random.shuffle(img_tuple)

        img_tuple = [self.transforms(img[1]) for img in img_tuple]
        
        return img_tuple, label
       
       

