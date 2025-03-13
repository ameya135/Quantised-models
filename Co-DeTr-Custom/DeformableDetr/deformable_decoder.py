import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .deformable_attention import DeformableAttention


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, num_levels: int = 4):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = DeformableAttention(d_model, nhead, num_levels)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.relu
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.)
        nn.init.constant_(self.linear2.bias, 0.)
        
    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(
            query=tgt,
            reference_points=self.get_reference_points(memory),
            input_flatten=memory,
            input_spatial_shapes=self.get_spatial_shapes(memory),
            input_level_start_index=self.get_level_start_index(memory),
            input_padding_mask=memory_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward Network
        tgt = self.forward_ffn(tgt)
        
        return tgt
    
    @staticmethod
    def get_reference_points(src: torch.Tensor) -> torch.Tensor:
        # Same as encoder implementation
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
        # Same as encoder implementation
        B, H, W, C = src.shape
        return torch.as_tensor([[H, W]], device=src.device, dtype=torch.long)
    
    @staticmethod
    def get_level_start_index(src: torch.Tensor) -> torch.Tensor:
        # Same as encoder implementation
        B, H, W, C = src.shape
        return torch.as_tensor([0], device=src.device, dtype=torch.long)