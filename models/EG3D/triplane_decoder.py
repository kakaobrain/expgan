import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TriplaneDecoder(nn.Module):
    '''
    [Inputs]
        - triplane => (B, triplane_channels, triplane_H, triplane_W)
        - points => (B, V * V, #step, 3)
    [outputs]
        - output: (B, V * V, #step, out_feature_channels + 1)
    '''
    def __init__(
        self,
        triplane_channels,
        out_feature_channels,
        hidden_channels,
        layers,
    ):
        super().__init__()

        in_channels = triplane_channels

        # build model
        model = [
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(inplace=True)
        ]

        for _ in range(layers - 2):
            model += [
                nn.Linear(hidden_channels, hidden_channels),
                nn.LeakyReLU(inplace=True)
            ]

        # TODO: softmax activation
        model += [nn.Linear(hidden_channels, out_feature_channels + 1)]

        self.model = nn.Sequential(*model)

    def forward(self, triplane, points):
        xy = points[..., 0:2]
        xz = torch.stack([points[..., 0], points[..., 2]], dim=-1)
        yz = points[..., 1:3]
        xy_xz_yz = torch.stack([xy, xz, yz], dim=1).view(-1, *xy.shape[1:])

        triplane_batch = triplane.view(triplane.shape[0] * 3, -1, *triplane.shape[-2:])
        in_feature = F.grid_sample(triplane_batch, xy_xz_yz, align_corners=False)
        in_feature = in_feature.view(triplane.shape[0], 3, -1, *in_feature.shape[-2:]).sum(dim=1).permute(0, 2, 3, 1)

        return self.model(in_feature)