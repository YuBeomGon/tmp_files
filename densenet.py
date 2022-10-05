import torch
import torchvision
from .common import *

# class CmapPafHeadAttention(torch.nn.Module):
#     def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
#         super(CmapPafHeadAttention, self).__init__()
#         self.cmap_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
#         self.paf_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
#         self.cmap_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
#         self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
            
#         self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
#         self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, x):
#         xc = self.cmap_up(x)
#         xp = self.paf_up(x) 

#         ac = torch.sigmoid(self.cmap_att(torch.cat((xc, xp), 1)))
#         ap = torch.tanh(self.paf_att(xp))
        
#         return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)


class DenseNetBackbone(torch.nn.Module):
    
    def __init__(self, densenet):
        super(DenseNetBackbone, self).__init__()
        self.densenet = densenet
    
    def forward(self, x):
        x = self.densenet.features(x)
        return x
    
    
def _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        DenseNetBackbone(densenet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    
    
def _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        DenseNetBackbone(densenet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def densenet121_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet121(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1024, num_upsample, num_flat)


def densenet169_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet169(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1664, num_upsample, num_flat)


def densenet201_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet201(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 1920, num_upsample, num_flat)


def densenet161_baseline(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet161(pretrained=pretrained)
    return _densenet_pose(cmap_channels, paf_channels, upsample_channels, densenet, 2208, num_upsample, num_flat)


    
def densenet121_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet121(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1024, num_upsample, num_flat)


def densenet169_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet169(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1664, num_upsample, num_flat)


def densenet201_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet201(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 1920, num_upsample, num_flat)


def densenet161_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    densenet = torchvision.models.densenet161(pretrained=pretrained)
    return _densenet_pose_att(cmap_channels, paf_channels, upsample_channels, densenet, 2208, num_upsample, num_flat)


