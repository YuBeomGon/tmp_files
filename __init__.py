from .resnet import *
from .densenet import *
from .efficientnet import *
from .mnasnet import *
# from .dla import *
from .swin import *
# from .swin_v1 import *
from .swin_mlp import *

MODELS = {
    'resnet18_baseline': resnet18_baseline,
    'resnet34_baseline': resnet34_baseline,
    'resnet50_baseline': resnet50_baseline,
    'resnet101_baseline': resnet101_baseline,
    'resnet152_baseline': resnet152_baseline,
    'resnet18_baseline_att': resnet18_baseline_att,
    'resnet34_baseline_att': resnet34_baseline_att,
    'resnet50_baseline_att': resnet50_baseline_att,
    'resnet101_baseline_att': resnet101_baseline_att,
    'resnet152_baseline_att': resnet152_baseline_att,
    'densenet121_baseline': densenet121_baseline,
    'densenet169_baseline': densenet169_baseline,
    'densenet201_baseline': densenet201_baseline,
    'densenet161_baseline': densenet161_baseline,
    'densenet121_baseline_att': densenet121_baseline_att,
    'densenet169_baseline_att': densenet169_baseline_att,
    'densenet201_baseline_att': densenet201_baseline_att,
    'densenet161_baseline_att': densenet161_baseline_att,
    'efficientnet_b0_baseline_att': efficientnet_b0_baseline_att,
    'efficientnet_b1_baseline_att': efficientnet_b1_baseline_att,
    'efficientnet_b2_baseline_att': efficientnet_b2_baseline_att,
    'efficientnet_b3_baseline_att': efficientnet_b3_baseline_att,
    'efficientnet_b4_baseline_att': efficientnet_b4_baseline_att,    
#     'dla34up_pose': dla34up_pose,
#     'dla60up_pose': dla60up_pose,
#     'dla102up_pose': dla102up_pose,
#     'dla169up_pose': dla169up_pose,
    'mnasnet0_5_baseline_att': mnasnet0_5_baseline_att,
    'mnasnet0_75_baseline_att': mnasnet0_75_baseline_att,
    'mnasnet1_0_baseline_att': mnasnet1_0_baseline_att,
    'mnasnet1_3_baseline_att': mnasnet1_3_baseline_att,
    'swin_tiny_baseline_att' : swin_tiny_baseline_att,
#    'swin_tiny_baseline_att_v1':swin_tiny_baseline_att_v1,
    'swin_small_baseline_att' : swin_small_baseline_att,
    'swin_base_baseline_att' : swin_base_baseline_att,
    # 'swin_large_baseline_att' : swin_large_baseline_att,
    'swin_mlp_tiny_baseline_att' : swin_mlp_tiny_baseline_att,
    'swin_mlp_small_baseline_att' : swin_mlp_small_baseline_att,
}
