import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class ClipData(Dataset):
    def __init__(self,
        data_path = '/Users/zhangzongkun/Desktop/research/video ordering/code/dataset/',#to .../cataract-101/
        stage = "train",
        transforms = transforms.Compose([transforms.Resize(224),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    ): 
        super().__init__()
        self.stage = stage
        self.video_dir = data_path + 'videos/'
        self.annotation_df = pd.read_csv(data_path + 'annotations.csv', sep=';')
        self.clip_num = None
        self.clip_len = 256
        self.sample_len = 64
        self.transforms = transforms

    def __len__(self):
        if self.clip_num is not None:
            return self.clip_num['num'].sum()
        self.clip_num = pd.DataFrame(columns=['VideoID','num'])
        for idx in self.annotation_df['VideoID'].unique():
            start = self.annotation_df[self.annotation_df['VideoID'] == idx].iloc[0]['FrameNo']
            vid = cv2.VideoCapture(self.video_dir+f'case_{idx}.mp4')
            end = vid.get(7)
            num =  (end- start) // self.clip_len
            self.clip_num.loc[self.clip_num.shape[0]] = [idx, num]
        return self.clip_num['num'].sum()

    def __getitem__(self, index):
        vid = None
        clip2_indx = None
        for row in self.clip_num.iterrows():
            index -= row['num']
            if 0 > index:
                vid = row['VideoID']
                index += row['num']
            # randomly pick another clip from the same video
            while clip2_indx == index:
                clip2_indx = random.randint(0,row['num']-1)
        video = cv2.VideoCapture(self.video_dir+f'case_{vid}.mp4')
        first_frame = self.annotation_df[self.annotation_df['VideoID'] == vid].iloc[0]['FrameNo'] + index * self.clip_len
        first_frame2 = self.annotation_df[self.annotation_df['VideoID'] == vid].iloc[0]['FrameNo'] + clip2_indx * self.clip_len
        sample1 = self.get_clip(first_frame, video)
        sample2 = self.get_clip(first_frame2, video)
        sample1 = sample1.transpose((0, 3, 1, 2))
        sample2 = sample2.transpose((0, 3, 1, 2))
        label = [1 if index < clip2_indx else -1]
        return {'clip1': torch.from_numpy(sample1), 'clip2': torch.from_numpy(sample2), 'label': torch.FloatTensor(label)}
        
    def get_clip(self, first_frame, video):
        video.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        # sample sample_len frames in the following clip_len frames
        tmp_sample = np.zeros((self.sample_len, 224, 224, 3))
        for i in range(self.sample_len):
            ret, img = video.read()
            if not ret:
                print('fail to read frame')
                return
            tmp_sample[i,:,:,:] = self.transforms(img)
            video.set(cv2.CAP_PROP_POS_FRAMES, first_frame + (i + 1) * self.clip_len/self.sample_len)
        return tmp_sample



class Clip_data_npy(Dataset):
    def __init__(self,
        data_path = '/projects/skillvba/data/sortvideosintime/samples/',#to .../cataract-101/
        stage = "train",
        transforms = transforms.Compose([transforms.Resize(224),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    ): 
        super().__init__()
        self.stage = stage
        if self.stage == 'train':
            self.video_dir = data_path + 'train/'
        else:
            self.video_dir = data_path + 'test/'
        #self.annotation_df = pd.read_csv(data_path + 'annotations.csv', sep=';')
        self.clip_num = defaultdict(int)
        self.clip_len = 256
        self.sample_len = 64

        for file in os.listdir(self.video_dir):
            vid = file.split('_')[0]
            self.clip_num[vid] += 1
        #self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.video_dir))

    def __getitem__(self, index):
        files = os.listdir(self.video_dir)
        sample1_name = files[index]
        vid = sample1_name.split('_')[0]
        clip_id1 = int(sample1_name.split('_')[1].split('.')[0])
        clip_num = self.clip_num[vid]
        clip_id2 = random.randint(0,clip_num-1)
        while clip_id2 == clip_id1:
            clip_id2 = random.randint(0,clip_num-1)

        if clip_id2 > clip_id1:
            label = 1
        else:
            label = 0

        sample2_name = f'{vid}_{clip_id2}.npy'
        clip1, clip2 = np.load(self.video_dir+sample1_name), np.load(self.video_dir+sample2_name)
        from_2_take_1 = np.arange(0,clip1.shape[0], 2)
        clip1, clip2 = clip1[from_2_take_1, :], clip2[from_2_take_1, :]
        #clip1, clip2 = clip1[np.arange(0,32, 2), :], clip2[np.arange(0,32, 2), :]
        #clip1, clip2 = normalization(clip1), normalization(clip2)
        #print(clip1.shape)
        return torch.from_numpy(clip1), torch.from_numpy(clip2), label
        #return clip1, clip2, torch.tensor([label])


class Clip_data_infer(Clip_data_npy):
    def __init__(self, bound = 5,data_path='/projects/skillvba/data/sortvideosintime/samples/',
                  stage="train", transforms=transforms.Compose([transforms.Resize(224), 
                                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])):
        super().__init__(data_path, stage, transforms)
        self.bound = bound

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        files = os.listdir(self.video_dir)
        sample1_name = files[index]
        vid = sample1_name.split('_')[0]
        clip_id1 = int(sample1_name.split('_')[1].split('.')[0])
        clip_num = self.clip_num[vid]
        clip_id2 = random.randint(max(0, clip_id1 - self.bound),min(clip_num-1, clip_id1 + self.bound))
        while clip_id2 == clip_id1:
            clip_id2 = random.randint(max(0, clip_id1 - self.bound),min(clip_num-1, clip_id1 + self.bound))

        if clip_id2 > clip_id1:
            label = 1
        else:
            label = -1

        sample2_name = f'{vid}_{clip_id2}.npy'
        clip1, clip2 = np.load(self.video_dir+sample1_name), np.load(self.video_dir+sample2_name)
        from_2_take_1 = np.arange(0,clip1.shape[0], 2)
        clip1, clip2 = clip1[from_2_take_1, :], clip2[from_2_take_1, :]
        #clip1, clip2 = clip1[np.arange(0,32, 2), :], clip2[np.arange(0,32, 2), :]
        #clip1, clip2 = normalization(clip1), normalization(clip2)
        #print(clip1.shape)
        return torch.from_numpy(clip1), torch.from_numpy(clip2), torch.Tensor([label])


        

if __name__ == "__main__":
    dummy = Clip_data_npy()
    dummy.__getitem__(0)