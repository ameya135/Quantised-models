import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DeformableAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_levels: int, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, 
                input_flatten: torch.Tensor, input_spatial_shapes: torch.Tensor,
                input_level_start_index: torch.Tensor, input_padding_mask: Optional[torch.Tensor] = None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        value = self.value_proj(input_flatten)
        
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        value = value.view(N, Len_in, self.n_heads, self.head_dim)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # Calculate sampling locations
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        output = self.deformable_attention_core(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights)
        
        output = self.output_proj(output)
        return output
