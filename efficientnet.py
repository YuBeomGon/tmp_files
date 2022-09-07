import torch
import torch.nn as nn
from .common import *
import timm
import math

from itertools import repeat
import collections.abc


class EfficientNetBackbone(torch.nn.Module):
    
    def __init__(self, efficientnet):
        super(EfficientNetBackbone, self).__init__()
        self.efficientnet = efficientnet
    
    def forward(self, x):
        x = self.efficientnet.forward_features(x)
        return x


def _efficientnet_pose(cmap_channels, paf_channels, upsample_channels, efficientnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        EfficientNetBackbone(efficientnet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

       
def _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        EfficientNetBackbone(efficientnet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model 

def efficientnet_b0_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    efficientnet = timm.create_model('efficientnet_b0', pretrained=pretrained)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 1280, num_upsample, num_flat)


def efficientnet_b1_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    efficientnet = timm.create_model('efficientnet_b1', pretrained=pretrained)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 1280, num_upsample, num_flat)


def efficientnet_b2_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    efficientnet = timm.create_model('efficientnet_b2', pretrained=pretrained)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 1408, num_upsample, num_flat)


def efficientnet_b3_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    efficientnet = timm.create_model('efficientnet_b3', pretrained=pretrained)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 1536, num_upsample, num_flat)


def efficientnet_b4_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    efficientnet = timm.create_model('efficientnet_b4', pretrained=pretrained)
    return _efficientnet_pose_att(cmap_channels, paf_channels, upsample_channels, efficientnet, 1792, num_upsample, num_flat)