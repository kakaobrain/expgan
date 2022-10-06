import torch
import torch.nn as nn

from .model import EqualLinear, PixelNorm


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MappingNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.layers = nn.Sequential(
            PixelNorm(),
            EqualLinear(input_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(hidden_dim, out_dim, lr_mul=0.01),
        )

    def forward(self, x):
        return self.layers(x)
