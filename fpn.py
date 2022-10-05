# https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.ops.misc import Conv2dNormActivation


class FeaturePyramidNetwork(nn.Module):
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()    
        self.inner_blocks = nn.ModuleList() # 1x1 conv
        self.layer_blocks = nn.ModuleList() # upsampling
        
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None
            )
            # layer_block_module = Conv2dNormActivation(
            #     out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None
            # )
            self.inner_blocks.append(inner_block_module)
            # self.layer_blocks.append(layer_block_module)        
            
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)            
                    
    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out           
    
    def forward(self, x: Tuple[str, Tensor]) -> Tensor:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        x = list(x)

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down

        return last_inner    