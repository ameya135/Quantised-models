import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .deformable_attention import DeformableAttention


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, num_levels: int = 4):
        super().__init__()
        
        self.self_attn = DeformableAttention(d_model, nhead, num_levels)
        
        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.relu
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)
        
    def forward_ffn(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2 = self.self_attn(
            query=src,
            reference_points=self.get_reference_points(src),
            input_flatten=src,
            input_spatial_shapes=self.get_spatial_shapes(src),
            input_level_start_index=self.get_level_start_index(src),
            input_padding_mask=src_mask
        )
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-forward Network
        src = self.forward_ffn(src)
        
        return src
    
    @staticmethod
    def get_reference_points(src: torch.Tensor) -> torch.Tensor:
        # Implementation depends on your specific needs
        # This is a simplified version
        B, H, W, C = src.shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=src.device),
            torch.linspace(0.5, W - 0.5, W, device=src.device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 0] = ref[..., 0] / W
        ref[..., 1] = ref[..., 1] / H
        return ref.reshape(B, H*W, 2)
    
    @staticmethod
    def get_spatial_shapes(src: torch.Tensor) -> torch.Tensor:
        # Implementation depends on your specific needs
        # This is a simplified version
        B, H, W, C = src.shape
        return torch.as_tensor([[H, W]], device=src.device, dtype=torch.long)
    
    @staticmethod
    def get_level_start_index(src: torch.Tensor) -> torch.Tensor:
        # Implementation depends on your specific needs
        # This is a simplified version
        B, H, W, C = src.shape
        return torch.as_tensor([0], device=src.device, dtype=torch.long)