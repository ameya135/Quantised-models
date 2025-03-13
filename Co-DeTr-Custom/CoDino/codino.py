import torch
import torch.nn as nn
from typing import Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.sfp_neck import SFPNeck
from utils.mlp import MLP
from utils.vit_backbone import ViTBackbone
from .codino_transformer import CoDinoTransformer

class CoDINO(nn.Module):
    def __init__(
        self,
        num_classes: int = 1203,
        num_queries: int = 900,
        backbone_cfg: dict = None,
        neck_cfg: dict = None,
        transformer_cfg: dict = None
    ):
        super().__init__()
        
        # Build backbone
        self.backbone = ViTBackbone(**backbone_cfg)
        
        # Build neck
        self.neck = SFPNeck(**neck_cfg)
        
        # Build transformer
        self.transformer = CoDinoTransformer(**transformer_cfg)
        
        # Build prediction heads
        self.class_embed = nn.Linear(256, num_classes)
        self.bbox_embed = MLP(256, 256, 4, 3)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Build multi-scale features
        multi_scale_features = self.neck([features])
        
        # Transform features
        hs, memory = self.transformer(multi_scale_features[-1])
        
        # Predict classes and boxes
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'aux_outputs': [{
                'pred_logits': outputs_class[i],
                'pred_boxes': outputs_coord[i]
            } for i in range(len(outputs_class)-1)]
        }


