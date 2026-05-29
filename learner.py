import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_


from core import ForDiagnosis


class WATT(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = self.get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def get_relative_positions(self, window_size):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_positions = relative_positions.permute(1, 2, 0).contiguous()
        relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
        return relative_positions_log

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class LayNormal(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(LayNormal, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized_input = (input - mean) / std
        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)
        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Att(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WATT(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            if w_shape[0] == self.dim * 2:
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)

            attn_windows = self.attn(qkv)

            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)
            shifted_out = shifted_out.permute(0, 3, 1, 2)

            if self.shift_size > 0:
                out = shifted_out[:, :, self.shift_size:(self.shift_size + H),
                self.shift_size:(self.shift_size + W)]
            else:
                out = shifted_out[:, :, :H, :W]

            attn_out = self.proj(out)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = attn_out + conv_out
            else:
                out = attn_out

        elif self.conv_type in ['Conv', 'DWConv']:
            out = self.conv(V)
            out = self.proj(out)

        else:
            out = X

        return out



class Learner(nn.Module):
    def __init__(self, config, imgc, imgsz):
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        self.se1 = MK_ECA_Net(32)
        self.legm_modules = nn.ModuleDict()
        self.modules = nn.ModuleDict()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                continue

            elif name == 'csssm':
                channels = param[0]
                num_blocks = param[1] if len(param) > 1 else 2
                ssm_d_state = param[2] if len(param) > 2 else 16
                ssm_ratio = param[3] if len(param) > 3 else 2.0
                mlp_ratio = param[4] if len(param) > 4 else 4.0
                use_light = param[5] if len(param) > 5 else False

                module_name = f'csssm_{i}'

                self.modules[module_name] = VMambaForDiagnosis(
                        in_channels=channels,
                        out_channels=channels,
                        num_blocks=num_blocks,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        mlp_ratio=mlp_ratio,
                    )
                continue

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'se']:
                continue
            else:
                raise NotImplementedError(f"Layer type {name} not implemented")

    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' % \
                      (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
            elif name == 'csssm':
                tmp = 'csssm:(channels:%d, blocks:%d, d_state:%d)' % \
                      (param[0], param[1] if len(param) > 1 else 2, param[2] if len(param) > 2 else 16)
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'bn', 'se', 'max_pool2d', 'avg_pool2d']:
                tmp = name + ':' + str(tuple(param)) if param else name
                info += tmp + '\n'
        return info

    def forward(self, x, vars=None, bn_training=True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'reshape':
                x = x.view(x.size(0), *param)

            elif name == 'relu':
                x = F.relu(x, inplace=param[0])

            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name == 'tanh':
                x = F.tanh(x)

            elif name == 'sigmoid':
                x = torch.sigmoid(x)

            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            elif name == 'se':
                x = self.se1(x)

            elif name == 'csssm':
                module_name = f'csssm_{i}'
                x = self.vmamba_modules[module_name](x)

            else:
                raise NotImplementedError(f"Forward pass for {name} not implemented")

        assert idx == len(vars), f"Index mismatch: {idx} vs {len(vars)}"
        assert bn_idx == len(self.vars_bn), f"BN index mismatch: {bn_idx} vs {len(self.vars_bn)}"

        return x

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
