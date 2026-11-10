"""Backbones trích trực tiếp từ PalmBridge cell 2 của file người dùng.

Hai kiến trúc có tên param khác nhau. Loader xác định loại bằng checkpoint.
Không thay đổi tên tầng, để tương thích state_dict gốc.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FEATURE_DIM

class LearnableGaborLayer(nn.Module):
    def __init__(self):
        super().__init__()

        n = 32
        ks = 15

        self.theta = nn.Parameter(
            torch.linspace(0.0, math.pi, n + 1)[:-1]
        )

        self.sigma = nn.Parameter(
            torch.full((n,), 3.0)
        )

        self.lambd = nn.Parameter(
            torch.full((n,), 6.0)
        )

        self.psi = nn.Parameter(
            torch.zeros(n)
        )

        self.gamma = nn.Parameter(
            torch.full((n,), 0.5)
        )

        half = ks // 2

        ys = torch.arange(
            -half,
            half + 1,
            dtype=torch.float32,
        )

        xs = torch.arange(
            -half,
            half + 1,
            dtype=torch.float32,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        self.register_buffer(
            "xx",
            xx.contiguous(),
        )

        self.register_buffer(
            "yy",
            yy.contiguous(),
        )

        self.ks = ks

    def _build_filters(self):
        theta = self.theta

        sigma = (
            self.sigma.abs().clamp(min=0.5)
        )

        lambd = (
            self.lambd.abs().clamp(min=1.0)
        )

        psi = self.psi

        gamma = (
            self.gamma.abs().clamp(min=0.1)
        )

        xx = self.xx.unsqueeze(0)
        yy = self.yy.unsqueeze(0)

        cos_t = torch.cos(
            theta
        ).view(-1, 1, 1)

        sin_t = torch.sin(
            theta
        ).view(-1, 1, 1)

        sigma = sigma.view(
            -1, 1, 1
        )

        lambd = lambd.view(
            -1, 1, 1
        )

        psi = psi.view(
            -1, 1, 1
        )

        gamma = gamma.view(
            -1, 1, 1
        )

        x_rot = (
            xx * cos_t
            + yy * sin_t
        )

        y_rot = (
            -xx * sin_t
            + yy * cos_t
        )

        envelope = torch.exp(
            -(
                x_rot ** 2
                + gamma ** 2 * y_rot ** 2
            )
            / (2.0 * sigma ** 2)
        )

        kernel = (
            envelope
            * torch.cos(
                2.0 * math.pi * x_rot / lambd
                + psi
            )
        )

        kernel = (
            kernel
            - kernel.mean(
                dim=(1, 2),
                keepdim=True,
            )
        )

        return (
            kernel
            .unsqueeze(1)
            .contiguous()
        )

    def forward(self, x):
        filters = self._build_filters()

        return F.conv2d(
            x,
            filters,
            padding=self.ks // 2,
        )


class CompetitivePool(nn.Module):
    def forward(self, x):
        out, _ = x.abs().max(
            dim=1,
            keepdim=True,
        )
        return out


class CustomCompNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.gabor = LearnableGaborLayer()
        self.compete = CompetitivePool()
        self.gbn = nn.BatchNorm2d(1)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    cout,
                    cout,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.block1 = block(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = block(128, 256)

        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                256 * 4 * 4,
                FEATURE_DIM,
                bias=False,
            ),
            nn.BatchNorm1d(FEATURE_DIM),
        )

    def getFeatureCode(self, x):
        g = F.relu(self.gabor(x))
        g = self.gbn(self.compete(g))

        f = self.pool1(self.block1(g))
        f = self.pool2(self.block2(f))
        f = self.pool3(self.block3(f))

        f = self.gap(self.block4(f))
        z = self.embed(f)

        return F.normalize(z, p=2, dim=1)

    def forward(self, x):
        return self.getFeatureCode(x)


class OfficialGaborConv2d(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        kernel_size,
        stride=1,
        padding=0,
        init_ratio=1,
    ):
        super().__init__()

        if channel_in != 1:
            raise ValueError(
                "Official CompNet GaborConv2d only supports channel_in=1."
            )

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio
        self.kernel = 0

        if init_ratio <= 0:
            init_ratio = 1.0

        self._SIGMA = 9.2 * init_ratio
        self._FREQ = 0.057 / init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(
            torch.FloatTensor([self._GAMMA]),
            requires_grad=True,
        )

        self.sigma = nn.Parameter(
            torch.FloatTensor([self._SIGMA]),
            requires_grad=True,
        )

        self.theta = nn.Parameter(
            torch.FloatTensor(
                torch.arange(0, channel_out).float()
            )
            * math.pi
            / channel_out,
            requires_grad=False,
        )

        self.f = nn.Parameter(
            torch.FloatTensor([self._FREQ]),
            requires_grad=True,
        )

        self.psi = nn.Parameter(
            torch.FloatTensor([0]),
            requires_grad=False,
        )

    def genGaborBank(
        self,
        kernel_size,
        channel_in,
        channel_out,
        sigma,
        gamma,
        theta,
        f,
        psi,
    ):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1

        y_0 = torch.arange(
            ymin,
            ymax + 1,
        ).float()

        x_0 = torch.arange(
            xmin,
            xmax + 1,
        ).float()

        y = (
            y_0
            .view(1, -1)
            .repeat(
                channel_out,
                channel_in,
                ksize,
                1,
            )
        )

        x = (
            x_0
            .view(-1, 1)
            .repeat(
                channel_out,
                channel_in,
                1,
                ksize,
            )
        )

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = (
            x
            * torch.cos(
                theta.view(-1, 1, 1, 1)
            )
            + y
            * torch.sin(
                theta.view(-1, 1, 1, 1)
            )
        )

        y_theta = (
            -x
            * torch.sin(
                theta.view(-1, 1, 1, 1)
            )
            + y
            * torch.cos(
                theta.view(-1, 1, 1, 1)
            )
        )

        gb = -torch.exp(
            -0.5
            * (
                (gamma * x_theta) ** 2
                + y_theta ** 2
            )
            / (
                8
                * sigma.view(-1, 1, 1, 1) ** 2
            )
        ) * torch.cos(
            2
            * math.pi
            * f.view(-1, 1, 1, 1)
            * x_theta
            + psi.view(-1, 1, 1, 1)
        )

        gb = (
            gb
            - gb.mean(
                dim=[2, 3],
                keepdim=True,
            )
        )

        return gb

    def forward(self, x):
        kernel = self.genGaborBank(
            self.kernel_size,
            self.channel_in,
            self.channel_out,
            self.sigma,
            self.gamma,
            self.theta,
            self.f,
            self.psi,
        )

        self.kernel = kernel

        return F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=self.padding,
        )


class OfficialCompetitiveBlock(nn.Module):
    def __init__(
        self,
        channel_in,
        n_competitor,
        ksize,
        stride,
        padding,
        init_ratio=1,
        o1=32,
        o2=12,
    ):
        super().__init__()

        if channel_in != 1:
            raise ValueError(
                "Official CompNet block only supports channel_in=1."
            )

        self.channel_in = 1
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        self.gabor_conv2d = OfficialGaborConv2d(
            channel_in=1,
            channel_out=n_competitor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            init_ratio=init_ratio,
        )

        self.a = nn.Parameter(
            torch.FloatTensor([1])
        )

        self.b = nn.Parameter(
            torch.FloatTensor([0])
        )

        self.argmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(
            n_competitor,
            o1,
            5,
            1,
            0,
        )

        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            o1,
            o2,
            1,
            1,
            0,
        )

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x = (x - self.b) * self.a
        x = self.argmax(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


class OfficialCompNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.cb1 = OfficialCompetitiveBlock(
            channel_in=1,
            n_competitor=9,
            ksize=35,
            stride=3,
            padding=0,
            init_ratio=1,
        )

        self.cb2 = OfficialCompetitiveBlock(
            channel_in=1,
            n_competitor=9,
            ksize=17,
            stride=3,
            padding=0,
            init_ratio=0.5,
        )

        self.cb3 = OfficialCompetitiveBlock(
            channel_in=1,
            n_competitor=9,
            ksize=7,
            stride=3,
            padding=0,
            init_ratio=0.25,
        )

        self.fc = nn.Linear(
            9708,
            512,
        )

        self.drop = nn.Dropout(p=0.25)

    def getFeatureCode(self, x):
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)

        x = torch.cat(
            (x1, x2, x3),
            dim=1,
        )

        x = self.fc(x)

        return F.normalize(
            x,
            p=2,
            dim=1,
        )

    def forward(self, x):
        return self.getFeatureCode(x)
