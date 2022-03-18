import torch
import self_cuda_backend as _ext
import torch.nn as nn
from math import sqrt
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import numpy as np


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class Relevance_Measuring(autograd.Function):
    @staticmethod
    def forward(ctx, query, key, radius=1, dilation=1):
        ctx.radius = radius
        ctx.dilation = dilation

        b, t, c, h, w = query.shape
        local_size = 2 * radius + 1
        size = (b, t, local_size * local_size * t, h, w)
        weight = torch.zeros(size, dtype=query.dtype, layout=query.layout, device=query.device)
        weight.fill_(-np.inf)
        _ext.weight_forward(query, key, weight, radius, dilation)
        ctx.save_for_backward(query, key)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        query, key = ctx.saved_tensors
        dquery = torch.zeros_like(query)
        dkey = torch.zeros_like(key)
        _ext.weight_backward(dw.contiguous(), query, key, dquery, dkey, ctx.radius, ctx.dilation)
        _check_contiguous(dquery, dkey)
        return dquery, dkey, None, None


class Spatial_Temporal_Aggregation(autograd.Function):
    @staticmethod
    def forward(ctx, weight, proj, radius=1, dilation=1):
        ctx.radius = radius
        ctx.dilation = dilation
        out = torch.zeros_like(proj)
        _ext.map_forward(weight, proj, out, radius, dilation)
        ctx.save_for_backward(weight, proj)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, proj = ctx.saved_tensors
        dweight = torch.zeros_like(weight)
        dproj = torch.zeros_like(proj)
        _ext.map_backward(dout.contiguous(), weight, proj, dweight, dproj, ctx.radius, ctx.dilation)
        _check_contiguous(dweight, dproj)
        return dweight, dproj, None, None


relevance_measuring = Relevance_Measuring.apply
spatial_temporal_aggregation = Spatial_Temporal_Aggregation.apply


class NS_Block(nn.Module):
    def __init__(self, channels_in=32, n_head=4, d_k=8, d_v=8, radius=[3, 3, 3, 3], dilation=[1, 3, 5, 7]):
        super(NS_Block, self).__init__()
        self.channels_in = channels_in
        self.n_head = n_head
        self.d_k = d_k
        self.radius = radius
        self.dilation = dilation
        self.query_conv = nn.Conv3d(channels_in, n_head * d_k, 1, bias=False)
        self.key_conv = nn.Conv3d(channels_in, n_head * d_k, 1, bias=False)
        self.value_conv = nn.Conv3d(channels_in, n_head * d_v, 1, bias=False)
        self.output_Linear = nn.Conv3d(n_head * d_v, channels_in, 1, bias=False)
        self.bn = nn.LayerNorm([int(self.channels_in/self.n_head), 16, 28])

    def forward(self, first, x):
        dilation, radius = self.dilation, self.radius
        x_ = x.permute(0, 2, 1, 3, 4).contiguous()
        first_ = first.permute(0, 2, 1, 3, 4).contiguous()
        query = self.query_conv(first_).permute(0, 2, 1, 3, 4)
        query_chunk = query.chunk(self.n_head, 2)
        key = self.key_conv(x_).permute(0, 2, 1, 3, 4)
        key_chunk = key.chunk(self.n_head, 2)
        value = self.value_conv(x_).permute(0, 2, 1, 3, 4)
        value_chunk = value.chunk(self.n_head, 2)

        M_T, M_A = [], []

        for i in range(self.n_head):
            query_i = query_chunk[i].contiguous()
            query_i = self.bn(query_i)
            key_i = key_chunk[i].contiguous()
            value_i = value_chunk[i].contiguous()
            M_A_i = relevance_measuring(query_i, key_i, radius[i], dilation[i]) / sqrt(self.channels_in/self.n_head)
            M_A.append(F.softmax(M_A_i, dim=2))
            M_T.append(spatial_temporal_aggregation(M_A_i, value_i, radius[i], dilation[i]))

        M_S, _ = torch.max(torch.cat(M_A, dim=2), dim=2)
        M_T = torch.cat(M_T, dim=2).permute(0, 2, 1, 3, 4)
        out_cat = self.output_Linear(M_T) * M_S.unsqueeze(2).permute(0, 2, 1, 3, 4)

        return out_cat.permute(0, 2, 1, 3, 4)


__all__ = ["NS_Block", "relevance_measuring", "spatial_temporal_aggregation"]
