import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from DeformableDetr.deformable_decoder import DeformableTransformerDecoderLayer
from DeformableDetr.deformable_encoder import DeformableTransformerEncoderLayer


class CoDinoTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        num_feature_levels: int = 5,
        num_query: int = 900
    ):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_levels=num_feature_levels
            ) for _ in range(num_encoder_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_levels=num_feature_levels
            ) for _ in range(num_decoder_layers)
        ])
        
        self.query_embed = nn.Embedding(num_query, d_model)
        
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query_embed: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = src.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Encoder
        memory = src
        for layer in self.encoder:
            memory = layer(memory, src_mask=mask, pos=pos_embed)
            
        # Decoder
        tgt = torch.zeros_like(query_embed)
        for layer in self.decoder:
            tgt = layer(
                tgt,
                memory,
                memory_mask=mask,
                pos=pos_embed,
                query_pos=query_embed
            )
            
        return tgt, memory

