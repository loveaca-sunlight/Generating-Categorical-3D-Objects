import torch
import numpy as np
import torch.nn as nn
from kornia.filters import filter2D

class SuperResolution(torch.nn.Module):
    
    def __init__(self, channels, input_resolution = 64, img_resolution = 128):
        super().__init__()

        self.input_resolution = input_resolution
        self.block0 = SynthesisBlock(channels, 128, resolution=img_resolution, img_channels=3)
        #self.block1 = SynthesisBlock(128, 64, resolution=img_resolution, img_channels=3)

    def forward(self, rgb, x):
        batch = rgb.shape[0]
        if len(rgb.shape) < 4:
            rgb = rgb.reshape(batch, self.input_resolution, self.input_resolution, 3).permute(0,3,1,2)
            x = x.reshape(batch, self.input_resolution, self.input_resolution, 32).permute(0,3,1,2)
        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=True)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=True)

        x, rgb = self.block0(x, rgb)
        #x, rgb = self.block1(x, rgb)
        return rgb
    

class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        #self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        #上采样后再用一个conv2d进行高斯滤波，暂时先空着不做
        self.num_conv = 0
        self.num_torgb = 0

        self.conv0 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.num_conv += 1

        self.conv1 = ConvBlock(out_channels, out_channels, 3, 1, 1)
        self.num_conv += 1

        self.torgb = ConvBlock(out_channels, img_channels, 3, 1, 1)
        self.num_torgb += 1

        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

    def forward(self, x, img):
        # Main layers.
        x = self.Upsample(x)
        x = self.conv0(x)
        x = self.conv1(x)

        # ToRGB.
        if img is not None:
            #misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = self.Upsample_2(img)
        y = self.torgb(x)
        img = img.add_(y) if img is not None else y

        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2,inplace=True)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)
