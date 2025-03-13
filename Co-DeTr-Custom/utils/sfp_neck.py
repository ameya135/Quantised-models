import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SFPNeck(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        use_p2: bool = True
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[0], out_channels, 1)
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(num_outs)
        ])
        
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Implementation of Scale Feature Pyramid Network
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
            
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        
        return outs
