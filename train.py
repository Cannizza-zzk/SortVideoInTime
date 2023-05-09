from Net import Three_resnet50
from tupledata import tuple_dataset
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
import numpy as np
import random
import os


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
file =  open('/projects/skillvba/code/course_sortvideosintime/tuple_log.txt', 'w+')
sys.stdout = file  

 # Set hyperparameters
num_epochs = 30
batch_size = 8
learning_rate = 0.001

# Prepare the data
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = tuple_dataset(transforms=transform)
train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = tuple_dataset(transforms=transform, stage='test')
test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet50 model
model = Three_resnet50(num_classes=2)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('Inf')

# Train the model...
for epoch in range(num_epochs):
    #print('running')
    epoch_loss = 0
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        inputs[2] = inputs[2].to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        
    if loss.item() < best_loss and epoch > 5:
        best_loss = loss.item()
        torch.save(model.state_dict(),f'/projects/skillvba/code/course_sortvideosintime/param/tuple_param/{epoch}_best_param.pkl')
    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

print(f'Finished Training, Loss: {loss.item():.4f}')



model.eval()

image_preds_all = []
image_targets_all = []

for inputs, labels in test_loader:
    # Move input and label tensors to the device
    inputs[0] = inputs[0].to(device)
    inputs[1] = inputs[1].to(device)
    inputs[2] = inputs[2].to(device)
    
    labels = labels.to(device)
    
    image_preds = model(inputs)  
    
    image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
    image_targets_all += [labels.detach().cpu().numpy()]
    
    

image_preds_all = np.concatenate(image_preds_all)
image_targets_all = np.concatenate(image_targets_all)

acc = (image_preds_all==image_targets_all).mean()

print(f'test accuracy = {acc} ' )




file.close()
sys.stdout = savedStdout 