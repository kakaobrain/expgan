import math

import torch.nn as nn

from .model import StyledConv, ToRGB


class StyleRenderer(nn.Module):
    def __init__(self, z_dim=256, tex_dim=16, kernel_size=1, use_noise=True, channel_multiplier=2, in_size=64, out_size=256):
        super().__init__()

        self.out_size = out_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        self.n_layers = int(math.log(out_size / in_size, 2))
        in_size_log = math.log(in_size, 2)

        in_channel = channels[in_size]

        self.conv1 = StyledConv(
            tex_dim,
            in_channel,
            kernel_size,
            z_dim,
            upsample=False,
            blur_kernel=None,
            demodulate=True,
            use_noise=use_noise
        )

        for i in range(self.n_layers):
            out_channel = channels[2 ** (i + in_size_log+1)]

            self.convs.append(
                StyledConv(in_channel,
                           out_channel,
                           kernel_size,
                           z_dim,
                           upsample=True,
                           blur_kernel=[1, 3, 3, 1],
                           demodulate=True,
                           use_noise=use_noise)
            )
            self.convs.append(
                StyledConv(
                    out_channel,
                    out_channel,
                    kernel_size,
                    z_dim,
                    upsample=False,
                    blur_kernel=None,
                    demodulate=True,
                    use_noise=use_noise
                )
            )
            self.to_rgbs.append(
                ToRGB(out_channel, z_dim)
            )
            in_channel = out_channel

    def forward(self, inp, style, out_size=None):
        if out_size is None:
            out_size = self.out_size

        inp = self.conv1(inp, style)
        skip = None

        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            inp = conv1(inp, style)
            inp = conv2(inp, style)
            skip = to_rgb(inp, style, skip)
            if out_size == skip.shape[-1]:
                break

        return skip
