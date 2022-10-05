import torch
import torchvision
from .common import *
from torchvision.models.mobilenetv2 import InvertedResidual, MobileNet_V2_Weights, MobileNetV2

class ResNetBackbone(torch.nn.Module):
    
    def __init__(self, mobilenet):
        super(ResNetBackbone, self).__init__()
        self.mobilenet = mobilenet
    
    def forward(self, x):
        
#        x = self.mobilenet.conv1(x)
#        x = self.mobilenet.bn1(x)
#        x = self.mobilenet.relu(x)
#        x = self.mobilenet.maxpool(x)
#
#        x = self.mobilenet.layer1(x) # /4
#        x = self.mobilenet.layer2(x) # /8
#        x = self.mobilenet.layer3(x) # /16
#        x = self.mobilenet.layer4(x) # /32
        x = self.mobilenet.features(x)

        return x


def _mobilenet_pose(cmap_channels, paf_channels, upsample_channels, mobilenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(mobilenet),
        CmapPafHead(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model
    

def _mobilenet_pose_att(cmap_channels, paf_channels, upsample_channels, mobilenet, feature_channels, num_upsample, num_flat):
    model = torch.nn.Sequential(
        ResNetBackbone(mobilenet),
        CmapPafHeadAttention(feature_channels, cmap_channels, paf_channels, upsample_channels, num_upsample=num_upsample, num_flat=num_flat)
    )
    return model

    
def mobilenet_v2_baseline_att(cmap_channels, paf_channels, upsample_channels=256, pretrained=True, num_upsample=3, num_flat=0):
    # mobilenet = torchvision.models.MobileNetV2(weights=MobileNet_V2_Weights.DEFAULT)
    # mobilenet = torchvision.models.MobileNetV2()
    # mobilenet = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    mobilenet = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    return _mobilenet_pose_att(cmap_channels, paf_channels, upsample_channels, mobilenet, 1280, num_upsample, num_flat)
