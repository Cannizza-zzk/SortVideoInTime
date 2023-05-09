from timm.models.vision_transformer import VisionTransformer, _cfg
import torch
import torch.nn as nn

class embed_layer(nn.Module):
    def __init__(self, backbone, embed_len, embed_dim, img_size, batch_size) -> None:
        super().__init__()
        self.cnn_backbone = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=img_size, stride= img_size) \
                        if backbone is None else backbone
        
    def backbone_forward(self, x):
        x = self.cnn_backbone(x).squeeze()
        if len(x.size()) == 2:
            x = x[None,:,:]
        elif len(x.size()) == 1:
            x = x[None,None,:]
        return x

    def forward(self,x):
        # print(x.size())
        x = x.permute(1,0,2,3,4)
        x = x.float()
        # print(x.size())
        #self.embed_seq = torch.empty(self.embed_len, x.size(dim=1), self.embed_dim)
        embed_seq = self.backbone_forward(x[0,:,:,:,:])

        for i in range(1, x.size(dim=0)):
            embed_seq = torch.cat((embed_seq, self.backbone_forward(x[i,:,:,:,:])), dim=0)
        embed_seq = embed_seq.permute(1,0,2)
        # print(self.embed_seq.device)
        return embed_seq



class video_vit(VisionTransformer):
    def __init__(self,backbone=None, batch_size = 8,embed_len = 64,embed_dim = 768, img_size = 224,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.embed_len = embed_len
        self.cnn_backbone = backbone
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, self.embed_dim))
        self.embed_layer = embed_layer(backbone, embed_len, embed_dim, img_size, batch_size)
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.embed_layer(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        # x = self.pre_logits(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_features(x1)
        x2 = self.forward_features(x2)
        return x1, x2
    
class video_vit_con(VisionTransformer):
    def __init__(self,backbone=None, batch_size = 8,embed_len = 64,embed_dim = 768, img_size = 224,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.embed_len = embed_len
        self.cnn_backbone = backbone
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_len, self.embed_dim))
        self.embed_layer = embed_layer(backbone, embed_len, embed_dim, img_size, batch_size)
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.embed_layer(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        # x = self.pre_logits(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_features(x1)
        x2 = self.forward_features(x2)
        concatenated_feature = torch.cat((x1, x2), dim=1)
        out = self.classifier(concatenated_feature)

        return out