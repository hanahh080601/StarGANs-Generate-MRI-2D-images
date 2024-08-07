import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, upsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        if downsample:
            self.downsample = nn.AvgPool2d(2)
        else:
            self.downsample = None
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            self.upsample = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.downsample:
            actual_out = self.downsample(out)
        elif self.upsample:
            actual_out = self.upsample(out)
        else:
            actual_out = out
        return actual_out, out


class ResUnetGenerator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=4, repeat_num=6):
        super(ResUnetGenerator, self).__init__()
        self.down_block1 = ResBlock(3+c_dim, conv_dim, downsample=True)
        self.down_block2 = ResBlock(conv_dim, conv_dim*2, downsample=True)
        self.encoder = ResBlock(conv_dim*2, conv_dim*4, upsample=True)  
        self.up_block = ResBlock(conv_dim*4, conv_dim*2, upsample=True)    
        self.final_block = ResBlock(conv_dim*2, conv_dim)
        self.final_conv = nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        out, down1 = self.down_block1(x)
        out, down2 = self.down_block2(out)
        out, _ = self.encoder(out)
        out = torch.cat([out, down2], dim=1)
        out, _ = self.up_block(out)
        out = torch.cat([out, down1], dim=1)
        out, _ = self.final_block(out)
        out = self.final_conv(out)
        return out
    
class ResUnet(nn.Module):
    def __init__(self, conv_dim=64, c_dim=4, repeat_num=6):
        super(ResUnet, self).__init__()

        self.input = []
        self.input.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.input.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.input.append(nn.ReLU(inplace=True))
        self.input = nn.Sequential(*self.input)

        self.down_block1 = ResBlock(conv_dim, conv_dim*2, downsample=True)
        self.down_block2 = ResBlock(conv_dim*2, conv_dim*4, downsample=True)
        
        self.bottle_neck = []
        for i in range(repeat_num):
            self.bottle_neck.append(ResidualBlock(conv_dim*4, conv_dim*4))
        self.bottle_neck = nn.Sequential(*self.bottle_neck)
        
        self.encoder = ResBlock(conv_dim*4, conv_dim*2, upsample=True)  
        self.up_block1 = ResBlock(conv_dim*4 + conv_dim, conv_dim*2, upsample=True)
        self.up_block2 = ResBlock(conv_dim*2 + conv_dim, conv_dim)
        
        self.output = []
        self.output.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.output.append(nn.Tanh())
        self.output = nn.Sequential(*self.output)
    
    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.input(x)
        out, down1 = self.down_block1(x)
        # print(out.shape, down1.shape)
        out, down2 = self.down_block2(out)
        # print(out.shape, down2.shape)
        out = self.bottle_neck(out)
        # print(out.shape)
        out, _ = self.encoder(out)
        # print(out.shape)
        out = torch.cat([out, down2], dim=1)
        # print(out.shape)
        out, _ = self.up_block1(out)
        out = torch.cat([out, down1], dim=1)
        out, _ = self.up_block2(out)
        out = self.output(out)
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=4, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))