import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation)

        channels_ = 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)


class TTOA(nn.Module):
    def __init__(self, low_channels, high_channels, c_kernel=3, r_kernel=3, use_att=False, use_process=True):
        """
        :param low_channels: low_level feature channels
        :param high_channels: high_level feature channels
        :param c_kernel: colum dcn kernels kx1 just use k
        :param r_kernel: row dcn kernels 1xk just use k
        :param use_att: bools
        :param use_process: bools
        """
        super(TTOA, self).__init__()

        self.l_c = low_channels
        self.h_c = high_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.att = use_att
        self.non_local_att = nn.Conv2d
        if self.l_c == self.h_c:
            print("Channel checked!")
        else:
            raise ValueError("Low and Hih channels need to be the same!")
        self.dcn_row = DCN(self.l_c, self.h_c, kernel_size=(1, self.r_k), stride=1, padding=(0, self.r_k // 2))
        self.dcn_colum = DCN(self.l_c, self.h_c, kernel_size=(self.c_k, 1), stride=1, padding=(self.c_k // 2, 0))
        self.sigmoid = nn.Sigmoid()
        if self.att is True:
            self.csa = self.non_local_att(self.l_c, self.h_c, 1, 1, 0)
        else:
            self.csa = None
        if use_process is True:
            self.preprocess = nn.Sequential(
                nn.Conv2d(self.l_c, self.h_c // 2, 1, 1, 0), nn.Conv2d(self.h_c // 2, self.l_c, 1, 1, 0)
            )
        else:
            self.preprocess = None

    def forward(self, a_low, a_high):
        if self.preprocess is not None:
            a_low = self.preprocess(a_low)
            a_high = self.preprocess(a_high)
        else:
            a_low = a_low
            a_high = a_high

        a_low_c = self.dcn_colum(a_low)
        a_low_cw = self.sigmoid(a_low_c)
        a_low_cw = a_low_cw * a_high
        a_colum = a_low + a_low_cw

        a_low_r = self.dcn_row(a_low)
        a_low_rw = self.sigmoid(a_low_r)
        a_low_rw = a_low_rw * a_high
        a_row = a_low + a_low_rw

        if self.csa is not None:
            a_TTOA = self.csa(a_row + a_colum)
        else:
            a_TTOA = a_row + a_colum
        return a_TTOA
