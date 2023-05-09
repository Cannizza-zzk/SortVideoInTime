from ViT_var import video_vit
from clipdata import Clip_data_npy
import timm
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import numpy as np
import sys

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

savedStdout = sys.stdout  
file =  open('/projects/skillvba/code/course_sortvideosintime/model_structure.txt', 'w+')
sys.stdout = file  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

# Load the ResNet50 model
model = video_vit(embed_len=32)

model_dict = model.state_dict()  
for k, v in model_dict.items():
    print(k)

print('pretrain model:')
""" data = np.load('/projects/skillvba/code/course_sortvideosintime/param/vit_param/imagenet_pretrain_vit.npz')
lst = data.files
for item in lst:
    print(item) """
model = timm.create_model('vit_base_patch32_224', pretrained=True)
model_dict = model.state_dict()  
for k, v in model_dict.items():
    print(k)

torch.save(model.state_dict(),f'./param/vit_param/imagenet_pretrain_vit.pkl')