from ViT_var import video_vit,video_vit_con
from clipdata import Clip_data_npy
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter   


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

writer = SummaryWriter('/projects/skillvba/code/course_sortvideosintime/tensorboard_log')

savedStdout = sys.stdout  
file =  open('./log_con.txt', 'w+')
sys.stdout = file  


# Set hyperparameters
num_epochs = 6
batch_size = 4
learning_rate = 0.001

# Prepare the data
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = Clip_data_npy()
train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = Clip_data_npy(stage='test')
test_loader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)





# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

# Load the ResNet50 model
model = video_vit_con(embed_len=32)

# load pretrained weight
NET_PARAMS_PATH = '/projects/skillvba/code/course_sortvideosintime/param/vit_param/imagenet_pretrain_vit.pkl'
net_params = torch.load(NET_PARAMS_PATH)
 
model_dict = model.state_dict() 
pretrained_dict = {k: v for k, v in net_params.items() if k in model_dict and k.startswith('blocks')}

model_dict.update(pretrained_dict)

#model_dict = torch.load('/projects/skillvba/code/course_sortvideosintime/param/vit_param/5_param.pkl')




 

model.load_state_dict(model_dict)
# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)


# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
# criterion = torch.nn.MarginRankingLoss(margin=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_loss = float('inf')

for epoch in range(1, 1 + num_epochs):
    print('running')
    epoch_loss = 0
    for x1, x2, labels in train_loader:
        # Move input and label tensors to the device

        x1 = x1.to(device)
        x2 = x2.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        
        # Forward pass
        #feature1, feature2 = model(x1,x2)
        preds = model(x1,x2)
        loss = criterion(preds,labels)
        writer.add_scalar('loss', loss, epoch)
        

        # Backward pass
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
   
    # Print the loss for every epoch
    print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')
    
    torch.save(model.state_dict(),f'./param/vit_con_param/{epoch}_param.pkl')

torch.save(model.state_dict(),f'./param/vit_con_param/{num_epochs}_param.pkl')

print(f'Finished Training, Loss: {loss.item():.4f}')

model.eval()

image_preds_all = []
image_targets_all = []
def is_prior(x1, x2, device):
    pos, neg = torch.tensor([1]), torch.tensor([-1])
    pos, neg = pos.to(device), neg.to(device)
    loss1 = nn.MarginRankingLoss(x1, x2, pos)
    loss2 = nn.MarginRankingLoss(x1, x2, neg)
    return loss1.item() < loss2.item()

for x1, x2, labels in test_loader:
    # Move input and label tensors to the device
    x1 = x1.to(device)
    x2 = x2.to(device)
    labels = labels.to(device)
    
    labels = labels.to(device)
    
    x1, x2 = model(x1,x2)  
    if is_prior(x1,x2,device):
        image_preds = 1
    else:
        image_preds = -1
    
    image_preds_all.append(image_preds)
    image_targets_all += [labels.detach().cpu().numpy()]
    
    

image_targets_all = np.concatenate(image_targets_all)

acc = (image_preds_all==image_targets_all).mean()

print(f'test accuracy = {acc} ' )

file.close()
sys.stdout = savedStdout 