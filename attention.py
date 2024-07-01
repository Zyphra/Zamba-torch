from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import transformer_engine as te

import torch
from rotary import *
from enums import AttnMaskType

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_number, attn_mask_type=AttnMaskType.padding, **kwargs):
        super().__init__()
        assert config.hidden_size % config.num_mem_heads == 0
        self.linear_qkv = nn.Linear(2 * config.hidden_size, 6 * config.hidden_size, bias=config.add_bias_linear)
        self.linear_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=config.add_bias_linear)
        self.n_head = config.num_mem_heads
        self.n_embd = config.hidden_size * 2
        self.dpa = te.pytorch.DotProductAttention(num_attention_heads=16, kv_channels =90, attention_dropout=0.0, layer_number=layer_number, attn_mask_type="padding")
                
    def forward(self, hidden_states, attention_mask, key_value_states=None, inference_params=None, rotary_pos_emb=None):
            
            qkv_out = self.linear_qkv(hidden_states)
            qkv_out = qkv_out.permute(1,0,2)
            # TODO FIX
            self.num_query_groups_per_partition = 16
            self.num_attention_heads_per_partition = 16
            self.hidden_size_per_attention_head = 90
            new_tensor_shape = qkv_out.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head * 2
            ),
        )
            qkv_out = qkv_out.view(*new_tensor_shape)

            (query, key, value) = torch.split(
                qkv_out,
                [
                    (
                        self.num_attention_heads_per_partition
                        // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head * 2
                    ),
                    self.hidden_size_per_attention_head * 2,
                    self.hidden_size_per_attention_head * 2,
                ],
                dim=3,
            )
            query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head * 2)

            if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2
            
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                query = apply_rotary_pos_emb(query, q_pos_emb)
                key = apply_rotary_pos_emb(key, k_pos_emb)
                
            y = self.dpa(query,key,value)
            y = y.transpose(1, 2).contiguous().view(B, T, C) 
            y = self.linear_proj(y)
            return y
