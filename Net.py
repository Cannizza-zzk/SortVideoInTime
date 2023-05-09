import torchvision.models as models
import torch
import torch.nn as nn


class Three_resnet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(Three_resnet50, self).__init__()
        resnet_50_model = models.resnet50(pretrained=pretrained)
        resnet_layer = nn.Sequential(*list(resnet_50_model.children())[:-1])
        self.resnet = resnet_layer
        #print(self.resnet)
        fc_features = resnet_50_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(fc_features*3, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward_once(self, x):
        output = self.resnet(x)
        return output

    def forward(self, x):
        x1 = self.forward_once(x[0])
        x2 = self.forward_once(x[1])
        x3 = self.forward_once(x[2])
        x = torch.concat((x1, x2, x3),1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
