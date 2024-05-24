import math
from typing import Optional, Union
import re
from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("CAUSAL CONV1d SUCCESS")
except ImportError:
    print("CAUSAL CONV1D IMPORT FAILED")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    print("SELECTIVE SCAN SUCCESS")
except ImportError:
    print("SELECTIVE SCAN FAILED")
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    print("LAYERNORM UPDATE SUCCESS")
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("LAYERNORM UPDATE FAILED")


from mamba_layer import MambaLayer
from mamba_config import MambaConfig
from mlp import MLP
from attention import CausalSelfAttention
from switch_mlp import SwitchMLP
from rotary import RotaryEmbedding


class MambaBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            assert config.num_mem_heads == 0, 'config.num_mem_heads > 0 only supports fused_add_norm=False'

    def forward(
        self, hidden_states: Tensor,  from_tf: Optional[Tensor] = None, residual: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            if from_tf is not None:
                hidden_states = self.norm((residual + from_tf).to(dtype=self.norm.weight.dtype))
            else:
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class AttentionBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            assert config.num_mem_heads == 0, 'config.num_mem_heads > 0 only supports fused_add_norm=False'
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None
        
        self.rotary_pos_emb = RotaryEmbedding(
                config.kv_channels, rotary_percent=1.0, seq_len_interpolation_factor=None
            )

    def forward(
        self, hidden_states: Tensor, from_tf: Optional[Tensor] = None, residual: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            if from_tf is not None:
                hidden_states = self.norm((residual + from_tf).to(dtype=self.norm.weight.dtype))
            else:
                hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = hidden_states.transpose(0,1).contiguous()
        rotary_seq_len = hidden_states.shape[0]
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        hidden_states = self.mixer(hidden_states, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask, inference_params=inference_params)
        hidden_states = hidden_states.transpose(0,1)
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Memory_AttentionBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, residual_in_fp32=False, fused_add_norm=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(config)
        assert config.rms_norm, 'Memory_AttentionBlock only supports RMSNorm'
        self.norm = norm_cls(2 * config.hidden_size)
        self.fused_add_norm = fused_add_norm
        
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None


    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # We don't include the residual in this block.
        # The residual is taken care of in vBlock.
        # This approach is more convenient as input and output dimensions are different.
        #hidden_states = hidden_states.transpose(0,1).contiguous()
        hidden_states = self.mixer(hidden_states, rotary_pos_emb=None, attention_mask=attention_mask, inference_params=inference_params)
        #hidden_states = hidden_states.transpose(0,1)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    


class MoEBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, attention_mask=None, layer_idx=None
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config, layer_idx=layer_idx)
        if config.use_module_layernorm and not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, from_tf: Optional[Tensor] = None, inference_params=None, attention_mask=None
    ):

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if self.config.add_bias_linear:
            hidden_states = sum(self.mixer(hidden_states))
        else:
            hidden_states = self.mixer(hidden_states)
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class vBlock(nn.Module):
    def __init__(
        self, config, sa_cls, mlp_cls=None, norm_cls=nn.LayerNorm, residual_in_fp32=False, layer_idx=None
    ):
        super().__init__()
        self.use_mem_mlp = config.use_mem_mlp
        self.sa = Memory_AttentionBlock(config, mixer_cls=sa_cls, norm_cls=norm_cls, residual_in_fp32=config.residual_in_fp32)
        if config.use_mem_mlp:
            self.mlp = MoEBlock(config, mixer_cls=mlp_cls, norm_cls=norm_cls, residual_in_fp32=config.residual_in_fp32, layer_idx=-1)

    def forward(self, hidden_states, residual=None, x_orig=None, inference_params=None, attention_mask=None):
        x = hidden_states + residual if residual is not None else hidden_states
        x = x.to(self.sa.mixer.linear_qkv.weight.dtype)
        x_ = torch.concatenate([x, x_orig], dim=-1)
        x = self.sa(x_, inference_params=inference_params, attention_mask=attention_mask)
        if self.use_mem_mlp:
            x, residual = self.mlp(x)
        return x.to(self.sa.mixer.linear_qkv.weight.dtype)


def create_block(config, layer_idx):
    factory_kwargs = {}
    
    if layer_idx == -1:
        norm_cls = partial(RMSNorm, eps=config.layernorm_epsilon, dtype=torch.float32)
        sa_cls = partial(CausalSelfAttention, **factory_kwargs, layer_number=-1)
        mlp_cls = partial(MLP, layer_idx=layer_idx, **factory_kwargs)
        block = vBlock(
            config,
            sa_cls=sa_cls,
            mlp_cls=mlp_cls,
            norm_cls=norm_cls,
            residual_in_fp32=config.residual_in_fp32, 
            layer_idx=layer_idx
        )
    else: 
        norm_cls = partial(RMSNorm, eps=config.layernorm_epsilon, dtype=torch.float32)
        
        if (not config.layer_mapping) or config.layer_mapping[layer_idx-1][0] == 'r' or config.layer_mapping[layer_idx-1][0] == 'g':
            if (not config.layer_mapping) or len(config.layer_mapping[layer_idx-1]) == 1:
                mixer_cls = partial(MambaLayer, layer_idx=layer_idx, **factory_kwargs)
                block = MambaBlock(
                    config,
                    mixer_cls=mixer_cls,
                    norm_cls=norm_cls,
                    fused_add_norm=config.fused_add_norm,
                    residual_in_fp32=config.residual_in_fp32,
                )
            elif config.layer_mapping[layer_idx-1][0] == 'a':
                mixer_cls = partial(SelfAttention, layer_number=layer_idx, **factory_kwargs)
                block = AttentionBlock(
                    config,
                    mixer_cls=mixer_cls,
                    norm_cls=norm_cls,
                    fused_add_norm=config.fused_add_norm,
                    residual_in_fp32=config.residual_in_fp32,
                )
        block.layer_idx = layer_idx
    return block

class MambaDecoder(nn.Module):
    
    def __init__(
        self,
        config: MambaConfig,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
    ):
        super().__init__()

        self.config: MambaConfig = config

        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        
        self.input_tensor = None

        self.checkpoint_core_block = self.config.recompute_granularity == 'selective'

        self.layer_mapping = self.config.layer_mapping

        self._build_layers()

    def _build_layers(self):

        num_layers_to_build = len(self.layer_mapping)
        self.layers = torch.nn.ModuleList([create_block(self.config, i + 1) for i in range(num_layers_to_build)])
        if self.config.num_mem_heads > 0:
            self.block = create_block(self.config, layer_idx=-1)
            self.block_map = torch.nn.ModuleList([nn.Linear(self.config.hidden_size, self.config.hidden_size, bias = self.config.add_bias_linear) if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g') else nn.Identity() for i in range(self.config.num_layers)]) 
            
        if self.post_process and self.post_layer_norm:
            self.final_layernorm = RMSNorm(self.config.hidden_size, eps=self.config.layernorm_epsilon, dtype=torch.float32)


    def forward(self, hidden_states, residual = None, inference_params=None, attention_mask=None):

        if not self.pre_process:
            hidden_states = self.input_tensor

        residual = None
        x_orig = torch.clone(hidden_states)
        from_tf = None
        for i, layer in enumerate(self.layers):
            if self.config.num_mem_heads > 0:
                from_tf = self.block_map[i](
                    self.block(
                        hidden_states, residual, x_orig, inference_params=inference_params, attention_mask = attention_mask
                    )
                ) if (i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g') \
                else None #(None, None)
            #if i%2 == 1 if (self.layer_mapping is None) else self.layer_mapping[i] == 'g':

            hidden_states, residual = layer(
                hidden_states=hidden_states,
                from_tf=from_tf,
                residual = residual,
                inference_params=inference_params,
                attention_mask = attention_mask
            )
        
        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            if not self.config.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.final_layernorm(residual.to(dtype=self.final_layernorm.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.final_layernorm, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.final_layernorm.weight,
                    self.final_layernorm.bias,
                    eps=self.final_layernorm.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        return hidden_states

    # def sharded_state_dict(self, prefix=''):

    #     sharded_state_dict = {}

    #     layer_prefix = f'{prefix}layers.'
    #     for layer in self.layers:
    #         sharded_state_dict.update(layer.sharded_state_dict(prefix=layer_prefix))

    #     if self.post_process and self.post_layer_norm:
    #         state_dict = self.state_dict(keep_vars=True)

    #         tensor = state_dict['final_layernorm.weight']
    #         layer_name = f'{prefix}final_layernorm.weight'
    #         sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)

    #         # RMSNorm doesn't have bias.
    #         if 'final_layernorm.bias' in state_dict.keys():
    #             tensor = state_dict['final_layernorm.bias']
    #             layer_name = f'{prefix}final_layernorm.bias'
    #             sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(
    #                 tensor, layer_name
    #             )

    #     return sharded_state_dict
