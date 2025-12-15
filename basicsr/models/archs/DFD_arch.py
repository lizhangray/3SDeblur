import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class DFD(nn.Module):
    def __init__(self, n_feats=24, n_encoder_res=6):
        super(DFD, self).__init__()
        self.a = nn.Parameter(torch.Tensor([[1 / 2, 0, 1 / 2, 0, 1 / 2],
                                            [0, 0, 0, 0, 0],
                                            [1 / 2, 0, 0, 0, 1 / 2],
                                            [0, 0, 0, 0, 0],
                                            [1 / 2, 0, 1 / 2, 0, 1 / 2]]), requires_grad=False)
        self.b = nn.Parameter(torch.Tensor([[1 / 2, 0, 1 / 2, 0, 1 / 2],
                                            [0, 1, 1, 1, 0],
                                            [1 / 2, 1, 0, 1, 1 / 2],
                                            [0, 1, 1, 1, 0],
                                            [1 / 2, 0, 1 / 2, 0, 1 / 2]]), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor([[0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 1 * 8, 0, 0],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0], ]), requires_grad=False)
        E1 = [nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 4, n_feats * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 8, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 4, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)
        ]

        E4 = [nn.Conv2d(n_feats, 3, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True)]

        cbam = [CBAM(n_feats, 8) for i in range(1)]

        E = E1 + cbam + E2 + E3 + E4
        self.E = nn.Sequential(
            *E
        )
        self.v = nn.Parameter(torch.Tensor([0]), requires_grad=True)

        self.alpha = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

    def manual_conv2d(self, input_tensor, weight, bias=None, stride=1, padding=2, normalization=None):
        # 确保数据在 GPU 上
        device = input_tensor.device

        b, ic, ih, iw = input_tensor.shape
        _, _, kh, kw = weight.shape

        oh = ((ih - kh + 2 * padding) // stride) + 1
        ow = ((iw - kw + 2 * padding) // stride) + 1

        if padding > 0:
            input_tensor = F.pad(input_tensor, (padding, padding, padding, padding))

        # 初始化输出张量
        output_tensor = torch.zeros((b, ic, oh, ow), device=device)

        # 展开输入张量以便进行矩阵乘法
        unfolded_input = F.unfold(input_tensor, kernel_size=(kh, kw), stride=stride, padding=0)

        # 展开后的输入张量形状: (b, kh*kw, oh*ow)
        unfolded_input = unfolded_input.view(b * ic, kh * kw, oh * ow)

        # 权重需要展开为矩阵形式: (oc, kh*kw)
        unfolded_weight = weight.view(1, 1, -1).to(device)

        # 矩阵乘法，结果的形状是 (b*ic, oc, oh*ow)
        output_unfolded = torch.matmul(unfolded_weight, unfolded_input)
        # 变形回原始形状
        output_tensor = output_unfolded.view(b, ic, oh, ow)

        if normalization is not None:
            out_max = torch.max(output_tensor)
            out_min = torch.min(output_tensor)

            output_tensor = (output_tensor - out_min) / (out_max - out_min)

        # 添加偏置
        if bias is not None:
            bias = bias.view(1, -1, 1, 1)
            output_tensor += bias

        return output_tensor

    def forward(self, x):
        kernel = self.a * self.v * self.v - self.b * self.v + self.bias
        kernel = kernel.view(1, 1, 5, 5)
        identity = x
        res = self.manual_conv2d(identity, kernel, None, 1, 2)
        res = res * self.alpha

        fea = self.E(identity).squeeze(-1).squeeze(-1)

        return fea + res


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'min']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels * reduction_ratio, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(gate_channels * reduction_ratio, gate_channels, 1, 1, 0),
            nn.ReLU(),
            Flatten(),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'min':
                ix = -x
                imin_pool = F.lp_pool2d(ix, 2, (ix.size(2), ix.size(3)), stride=(ix.size(2), ix.size(3)))
                min_pool = -imin_pool
                channel_att_raw = self.mlp(min_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1), torch.min(x, 1)[0].unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'min'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
