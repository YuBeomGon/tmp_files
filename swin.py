import torch
import torch.nn as nn
from .common import *
import timm
import math

from itertools import repeat
import collections.abc

# # From PyTorch internals
# def _ntuple(n):
#     def parse(x)WindowProcessReverse:
#         if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
#             return x
#         return tuple(repeat(x, n))
#     return parse


# to_1tuple = _ntuple(1)
# to_2tuple = _ntuple(2)
# to_3tuple = _ntuple(3)
# to_4tuple = _ntuple(4)
# to_ntuple = _ntuple

# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
#         # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x


class SwinBackbone(torch.nn.Module):
    
    def __init__(self, swin):
        super(SwinBackbone, self).__init__()
        self.swin = swin
    
    def forward(self, x):
        x = self.swin.forward_features(x)
        B, L, C = x.shape
        L = int(math.sqrt(L))
        # x = x.view(B, L, L, C).permute(0,3,1,2)
        x = x.permute(0,2,1).contiguous().view(B,C,L,L)
        return x


def _swin_pose(cmap_channels, paf_channels, upsample_channels, swin, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        SwinBackbone(swin),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

       
def _swin_pose_att(cmap_channels, paf_channels, upsample_channels, swin, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        SwinBackbone(swin),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model 

def swin_tiny_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
    return _swin_pose_att(cmap_channels, paf_channels, upsample_channels, swin, 768, num_upsample, num_flat)


def swin_small_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    swin = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
    return _swin_pose_att(cmap_channels, paf_channels, upsample_channels, swin, 768, num_upsample, num_flat)


def swin_base_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    swin = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained)
    return _swin_pose_att(cmap_channels, paf_channels, upsample_channels, swin, 1024, num_upsample, num_flat)


def swin_large_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    swin = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
    return _swin_pose_att(cmap_channels, paf_channels, upsample_channels, swin, 1536, num_upsample, num_flat)
