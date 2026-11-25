"""Optional attention-kernel adapter; no AMB3R weights or geometry are replaced."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SDPA_CALLS = 0


def varlen_qkv_sdpa(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None):
    """PTv3's packed noncausal attention, preserving its exact patch boundaries.

    The caller uses only this subset of FlashAttention's API. Equal-length
    patches share one batched SDPA call; variable patches remain independent.
    """
    global SDPA_CALLS
    SDPA_CALLS += 1
    offsets = cu_seqlens.tolist()
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if not lengths or offsets[0] != 0 or offsets[-1] != len(qkv):
        raise ValueError('Invalid packed attention offsets')
    if min(lengths) <= 0 or max(lengths) > max_seqlen:
        raise ValueError('Invalid packed attention lengths')

    def attend(packed):
        # B,L,3,H,D -> B,H,L,D -> B,L,H,D
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=softmax_scale,
        ).transpose(1, 2)

    if len(set(lengths)) == 1:
        return attend(qkv.reshape(len(lengths), lengths[0], *qkv.shape[1:])).reshape(
            len(qkv), *qkv.shape[2:]
        )
    return torch.cat([
        attend(qkv[start:end].unsqueeze(0)).squeeze(0)
        for start, end in zip(offsets, offsets[1:])
    ], dim=0)


def enable_sdpa_attention():
    """Change only the PTv3 attention kernel in this process, before construction."""
    from ptv3 import point_transformer
    point_transformer.flash_attn = SimpleNamespace(
        flash_attn_varlen_qkvpacked_func=varlen_qkv_sdpa,
    )
    print('PTv3 attention: PyTorch SDPA with original packed patch boundaries', flush=True)
