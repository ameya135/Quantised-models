import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .transformer_block import TransformerBlock

class ViTBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 1536,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4 * 2/3,
        drop_path_rate: float = 0.3,
        window_size: int = 16,
        window_block_indexes: tuple = None
    ):
        super().__init__()
        # Define default window block indexes if not provided
        if window_block_indexes is None:
            self.window_block_indexes = tuple(
                list(range(0, 2)) + 
                list(range(3, 5)) + 
                list(range(6, 8)) + 
                list(range(9, 11)) + 
                list(range(12, 14)) + 
                list(range(15, 17)) + 
                list(range(18, 20)) + 
                list(range(21, 23))
            )
        else:
            self.window_block_indexes = window_block_indexes
            
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                window_size=window_size if i in self.window_block_indexes else 0
            ) for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)