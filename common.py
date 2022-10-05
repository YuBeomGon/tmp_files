import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class UpsampleCBR(torch.nn.Sequential):
    def __init__(self, input_channels, output_channels, count=1, num_flat=0):
        layers = []
        for i in range(count):
            if i == 0:
                inch = input_channels
            else:
                inch = output_channels
                
            layers += [
                torch.nn.ConvTranspose2d(inch, output_channels, kernel_size=4, stride=2, padding=1),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU()
            ]
            for i in range(num_flat):
                layers += [
                    torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.BatchNorm2d(output_channels),
                    torch.nn.ReLU()
                ]
            
        super(UpsampleCBR, self).__init__(*layers)

        
class SelectInput(torch.nn.Module):
    
    def __init__(self, index):
        super(SelectInput, self).__init__()
        self.index = index
    
    def forward(self, inputs):
        return inputs[self.index]
    
    
class CmapPafHead(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHead, self).__init__()
        if num_upsample > 0:
            self.cmap_conv = torch.nn.Sequential(
                UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat),
                torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
            )
            self.paf_conv = torch.nn.Sequential(
                UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat),
                torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.cmap_conv = torch.nn.Conv2d(input_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
            self.paf_conv = torch.nn.Conv2d(input_channels, paf_channels, kernel_size=1, stride=1, padding=0)
            
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)            
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        return self.cmap_conv(x), self.paf_conv(x)
    
    
class CmapPafHeadAttention(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafHeadAttention, self).__init__()
        self.cmap_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.paf_up = UpsampleCBR(input_channels, upsample_channels, num_upsample, num_flat)
        self.cmap_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
            
        self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
        
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)           
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        xc = self.cmap_up(x)
        ac = torch.sigmoid(self.cmap_att(xc))
        
        xp = self.paf_up(x)
        ap = torch.tanh(self.paf_att(xp))
        
        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)
    
class CmapPafConv(torch.nn.Module):
    def __init__(self, input_channels, cmap_channels, paf_channels, upsample_channels=256, num_upsample=0, num_flat=0):
        super(CmapPafConv, self).__init__()
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),            
        )
        
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),            
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),            
                        torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(upsample_channels),
                        torch.nn.ReLU(),            
        )        
        
        self.cmap_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
        self.paf_att = torch.nn.Conv2d(upsample_channels, upsample_channels, kernel_size=3, stride=1, padding=1)
            
        self.cmap_conv = torch.nn.Conv2d(upsample_channels, cmap_channels, kernel_size=1, stride=1, padding=0)
        self.paf_conv = torch.nn.Conv2d(upsample_channels, paf_channels, kernel_size=1, stride=1, padding=0)
        
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)           
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, x):
        xc = self.conv1(x)
        ac = torch.sigmoid(self.cmap_att(xc))
        
        xp = self.conv2(x)
        ap = torch.tanh(self.paf_att(xp))
        
        return self.cmap_conv(xc * ac), self.paf_conv(xp * ap)
