from ViT_var import video_vit
from clipdata import Clip_data_npy, Clip_data_infer
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

bound = 15
savedStdout = sys.stdout  
file =  open(f'/projects/skillvba/code/course_sortvideosintime/infer_log/train_inference_log_bound{bound}.txt', 'w+')
sys.stdout = file  

testset = Clip_data_infer(bound=bound,stage='train')
test_loader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

# Load the ResNet50 model
model = video_vit(batch_size=1, embed_len=32)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load('/projects/skillvba/code/course_sortvideosintime/param/vit_param/10_param.pkl'))


model = model.to(device)
model.eval()

image_preds_all = []
image_targets_all = []
def is_prior(x1, x2, device):
    pos, neg = torch.tensor([1]), torch.tensor([-1])
    pos, neg = pos.to(device), neg.to(device)
    pos, neg = pos.view([1,-1]), neg.view([1,-1])
    criterion = nn.MarginRankingLoss()
    loss1 = criterion(x1, x2, pos)
    loss2 = criterion(x1, x2, neg)
    return loss1.item() < loss2.item()

for x1, x2, labels in test_loader:
    # Move input and label tensors to the device
    x1 = x1.to(device)
    x2 = x2.to(device)
    labels = labels.to(device)
    
    labels = labels.to(device)
    
    feature1, feature2 = model(x1,x2)  
    """ feature1 = feature1.squeeze()
    feature2 = feature2.squeeze() """
    if is_prior(feature1,feature2,device):
        image_preds = 1
    else:
        image_preds = -1
    
    image_preds_all.append(image_preds)
    image_targets_all += [labels.detach().cpu().numpy()]
    
    

image_targets_all = np.concatenate(image_targets_all)
print('inference result:')

print(image_preds_all)

print('\n\n labels:')
print(image_targets_all.squeeze())

acc = (image_preds_all==image_targets_all).mean()

print(f'test accuracy = {acc} ' )


file.close()
sys.stdout = savedStdout 
