import torch

from .masked_drop import MaskedDrop
from .spatial_pool import SpatialPool
from .qformer import Qformer
from .vlm_attention import VlmAttention
from .perceiver import DynamicCompressor

class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}

def build_vision_resampler(model_args, delay_load=False, **kwargs):
    # import pdb;pdb.set_trace()
    resampler_type = getattr(model_args, 'mm_resampler_type', None)
    if resampler_type == 'masked_drop':
        return MaskedDrop(model_args)
    elif resampler_type == 'spatial_pool':
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == 'qformer':
        return Qformer(model_args, **kwargs)
    elif resampler_type == 'vlm_attention':
        return VlmAttention(model_args,**kwargs)
    elif resampler_type == 'dynamic_compressor':
        return DynamicCompressor(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()
    else:
        raise ValueError(f'Unknown resampler type: {resampler_type}')
