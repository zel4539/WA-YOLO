import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from functools import partial


# 创建小波滤波器
def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


# 小波变换
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


# 逆小波变换
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# 小波变换卷积 (WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 小波滤波器和逆滤波器
        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)

        # 检查并注册缓冲区
        if not hasattr(self, 'wt_filter'):
            self.register_buffer('wt_filter', wt_filter)
        if not hasattr(self, 'iwt_filter'):
            self.register_buffer('iwt_filter', iwt_filter)

        # 基础卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, groups=in_channels, bias=bias)
        self.base_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # 小波卷积
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, groups=in_channels * 4, bias=False)
            for _ in range(self.wt_levels)
        ])
        # 修复：确保每个元素是 nn.Parameter
        self.wavelet_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1, in_channels * 4, 1, 1) * 0.1) for _ in range(self.wt_levels)
        ])

        # 下采样
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=True)
            self.do_stride = lambda x: F.conv2d(x, self.stride_filter, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll = x
        for i in range(self.wt_levels):
            x_wt = wavelet_transform(x_ll, self.wt_filter)
            x_ll = x_wt[:, :, 0, :, :]
            x_wt = x_wt.reshape(x_wt.shape[0], x_wt.shape[1] * 4, x_wt.shape[3], x_wt.shape[4])
            x_wt = self.wavelet_convs[i](x_wt) * self.wavelet_scale[i]
            x_wt = x_wt.reshape(x_wt.shape[0], x_wt.shape[1] // 4, 4, x_wt.shape[2], x_wt.shape[3])
            x_ll = inverse_wavelet_transform(x_wt, self.iwt_filter)

        x_base = self.base_conv(x) * self.base_scale

        # 确保 x_ll 和 x_base 的尺寸一致
        if x_ll.shape[2:] != x_base.shape[2:]:
            x_ll = F.interpolate(x_ll, size=x_base.shape[2:], mode='bilinear', align_corners=False)

        x = x_base + x_ll

        if self.do_stride is not None:
            x = self.do_stride(x)
        return x


# SimAM 注意力机制
class SimAM(nn.Module):
    def __init__(self, channels):
        super(SimAM, self).__init__()
        self.channels = channels
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 1e-6)) + 0.5
        return x * (1 + torch.sigmoid(self.gamma * y))


class DynamicSelection(nn.Module):
    def __init__(self, channels):
        super(DynamicSelection, self).__init__()
        self.wtc = WTConv2d(channels, channels)
        self.simam = SimAM(channels)
        self.gate = nn.Conv2d(channels, 1, kernel_size=1)  # 动态选择门控

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))  # 生成选择权重
        wtc_out = self.wtc(x)
        simam_out = self.simam(x)
        return gate * wtc_out + (1 - gate) * simam_out  # 加权融合


# C3k 模块
class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = nn.Conv2d(c1, self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv2 = nn.Conv2d(c1, self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.m = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, groups=g, bias=False),
                nn.BatchNorm2d(self.c),
                nn.ReLU(inplace=True)
            ) for _ in range(n)
        ])
        self.cv3 = nn.Conv2d(self.c * 2, c2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m(self.cv2(x))
        return self.cv3(torch.cat([x1, x2], dim=1))


# C2f_WTC 类（带动态选择机制）
class C2f_WTC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.m = nn.ModuleList(nn.Sequential(
            nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(self.c),
            nn.ReLU(inplace=True)
        ) for _ in range(n))
        self.att = DynamicSelection(c2)  # 使用动态选择机制

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y_cat = torch.cat(y, 1)
        y_cat = self.cv2(y_cat)
        return self.att(y_cat)

# C3k2_WTC 类（带动态选择机制）
class C3k2_WTC(C2f_WTC):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else nn.Sequential(
                nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, groups=g, bias=False),
                nn.BatchNorm2d(self.c),
                nn.ReLU(inplace=True)
            ) for _ in range(n)
        )
        self.att = DynamicSelection(c2)  # 使用动态选择机制


# 测试代码
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)  # b c h w 输入
    model = C3k2_WTC(c1=32, c2=64, c3k=True)  # 启用 C3k 模块
    output = model(input)
    print(output.size())  # 预期输出: torch.Size([3, 64, 64, 64])