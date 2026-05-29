import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
import math


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = 0.1
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x = x.permute(0, 2, 3, 1).contiguous()

        x_flat = x.reshape(B, -1, self.d_inner)
        x_dbl = self.x_proj(x_flat)
        delta, B_ssm = x_dbl.chunk(2, dim=-1)

        A = -torch.exp(self.A_log.float())
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)

        y = x_flat * self.D + (deltaA * deltaB).sum(dim=-1)

        y = y.reshape(B, H, W, self.d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_conv: int = 3,
        ssm_drop_rate: float = 0.,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            d_conv=ssm_conv,
            expand=ssm_ratio,
            dropout=ssm_drop_rate,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp2D(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            drop=mlp_drop_rate,
        )

    def forward(self, input):
        B, C, H, W = input.shape
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x + self.drop_path(self.op(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Mlp2D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output



class ForDiagnosis(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_blocks=2,
        ssm_d_state=16,
        ssm_ratio=2.0,
        mlp_ratio=4.0,
        drop_path=0.1,
        use_cross_scan=True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.use_cross_scan = use_cross_scan
        self.in_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if use_cross_scan:
            self.cross_scan = CrossScan2D()
            self.cross_merge = CrossMerge2D()
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=out_channels,
                drop_path=drop_path,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

