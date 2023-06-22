import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import os


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
        
        
@register_model
def createModel(pretrained=True, preChannel=3, channel=3, preNumClasses=1000, numClasses=2, weightPath=None, device=None, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    print("pretrained:",pretrained)
    if pretrained:
        model.patch_embed.proj = nn.Conv2d(in_channels= preChannel, out_channels=768, kernel_size=(16, 16), stride=(16, 16)).cuda()
        model.head = nn.Linear(in_features=768, out_features=preNumClasses, bias=True).cuda()
        model.head_dist = nn.Linear(in_features=768, out_features=preNumClasses, bias=True).cuda()
        
        if torch.cuda.is_available():
            ckpt = torch.load(os.path.join(weightPath,"weight.pth"))
        else:
            ckpt = torch.load(os.path.join(weightPath,"weight.pth"), map_location=torch.device('cpu'))
        
        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key in ckpt:
                model_dict[key] = ckpt[key]
                
        model.load_state_dict(model_dict)
        
        for param in model.parameters():
            param.requires_grad = False

        model.patch_embed.proj = nn.Conv2d(in_channels=channel, out_channels=768, kernel_size=(16, 16), stride=(16, 16)).cuda()
        model.head = nn.Linear(in_features=768, out_features=numClasses, bias=True).cuda()
        model.head_dist = nn.Linear(in_features=768, out_features=numClasses, bias=True).cuda()

    else:
        model.patch_embed.proj = nn.Conv2d(in_channels=channel, out_channels=768, kernel_size=(16, 16), stride=(16,16)).cuda()
        model.head = nn.Linear(in_features=768, out_features=numClasses, bias=True).cuda()
        model.head_dist = nn.Linear(in_features=768, out_features=numClasses, bias=True).cuda()
    
    return model

if __name__ == "__main__":
   
    model = createModel(
        pretrained=True,
        channel=3,
        preNumClasses=1000,
        numClasses=1000,
        weightPath="/data/sungmin/Vgg19/originWeight",
        device="cuda"
    )
    print(model)
    
    import torch
    import torch.nn as nn
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    summary(model, input_size=(3, 224, 224))
