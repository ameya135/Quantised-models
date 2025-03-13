import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        drop_path: float = 0.,
        window_size: int = 0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop_path)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, dim, 2)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size > 0:
            x = self._window_attention(x)
        else:
            x = x + self.drop_path(self.attn(x, x, x)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        H = W = int(math.sqrt(x.shape[1]))
        x = x.view(-1, H, W, x.shape[-1])
        
        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            
        Hp, Wp = H + pad_h, W + pad_w
        
        # Window partition
        x = x.view(-1, Hp // self.window_size, self.window_size,
                  Wp // self.window_size, self.window_size, x.shape[-1])
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, x.shape[-1])
        
        # Window attention
        attn_windows = self.attn(windows, windows, windows)[0]
        
        # Reverse window partition
        x = attn_windows.view(-1, Hp // self.window_size, Wp // self.window_size,
                            self.window_size, self.window_size, x.shape[-1])
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, Hp, Wp, x.shape[-1])
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
            
        return x.view(-1, H * W, x.shape[-1])