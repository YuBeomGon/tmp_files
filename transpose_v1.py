from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List

import torch
import torch.nn as nn
from common import *
import timm
import math

from einops import rearrange
from einops.layers.torch import Rearrange

from itertools import repeat
import collections.abc

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)    
    
class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))
    
class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)  
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, mlp(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x    



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class PerPose(nn.Module):

    def __init__(self, block, layers, dim=256, mlp_dim=1024, depth=6, heads=8, img_size=(256,256), **kwargs):
        self.inplanes = 64

        super(PerPose, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        w, h = img_size

        self.reduce = nn.Conv2d(self.inplanes, dim, 1, bias=False)
        # self._make_position_embedding(w, h, dim, pos_embedding_type)
        self.abspos_embeding = nn.Parameter(torch.randn(w//8, h//8, dim)).view(1, -1, dim)
        self.heads = heads
        self.dim_head = dim // heads
        
        self.attn_layer = Transformer(dim, depth, heads, self.dim_head, mlp_dim)

        # used for deconv layers
        self.inplanes = dim
        
        self.apply(self._init_weights)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def forward_features(self, x) :
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.reduce(x)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x += self.abspos_embeding
        
        x = self.attn_layer(x)
        x = x.permute(0, 2, 1).contiguous().view(bs, c, h, w)   
        
        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       


class PerposeBackbone(torch.nn.Module):
    
    def __init__(self, perpose):
        super(PerposeBackbone, self).__init__()
        self.perpose = perpose
    
    def forward(self, x):
        x = self.perpose.forward_features(x)
        return x
    
    
def _perpose_pose(cmap_channels, paf_channels, upsample_channels, perpose, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        PerposeBackbone(perpose),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
     
def _perpose_pose_att(cmap_channels, paf_channels, upsample_channels, perpose, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        PerposeBackbone(perpose),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model 

def perpose_tiny_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=1, num_flat=0):
    # perpose = timm.create_model('perpose_tiny_patch4_window12_384', pretrained=pretrained)
    perpose = PerPose(Bottleneck, [3, 4],  dim=256, mlp_dim=1024, depth=6, 
                           heads=8, img_size=(384,384))
    return _perpose_pose_att(cmap_channels, paf_channels, upsample_channels, perpose, 256, num_upsample, num_flat)


def perpose_small_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=1, num_flat=0):
    perpose = PerPose(Bottleneck, [3, 4],  dim=256, mlp_dim=1024, depth=8, 
                           heads=8, img_size=(384,384))
    return _perpose_pose_att(cmap_channels, paf_channels, upsample_channels, perpose, 256, num_upsample, num_flat)


def perpose_base_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=1, num_flat=0):
    perpose = PerPose(Bottleneck, [3, 4],  dim=384, mlp_dim=1024, depth=12, 
                           heads=12, img_size=(384,384))
    return _perpose_pose_att(cmap_channels, paf_channels, upsample_channels, perpose, 384, num_upsample, num_flat)
