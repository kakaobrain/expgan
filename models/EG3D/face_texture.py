import torch
import torch.nn as nn
import torch.nn.functional as F
from models.StyleGAN2.model import EqualLinear


class FaceTexture(nn.Module):
    def __init__(self, in_channels, out_channels, volume_size):
        super().__init__()
        self.in_channels = in_channels
        self.face_texture_channels = out_channels
        self.volume_size = volume_size

        # pose encoder
        self.pose_encoder = nn.Sequential(
            EqualLinear(12, 32, lr_mul=0.01, activation="fused_lrelu"),
            EqualLinear(32, in_channels, lr_mul=0.01),
        )

        # last layer
        self.last = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels+1, kernel_size=3, stride=1, padding=1)  # rgb featrue + sigma
        )

    def forward(self, face_texture, uv, c2w=None):
        if c2w is not None:
            # pose condition to texture
            pose = c2w.reshape(-1, 12)
            pose = self.pose_encoder(pose)
            face_texture = self.last(face_texture + pose.view(-1, self.in_channels, 1, 1))
        else:
            face_texture = self.last(face_texture)

        self.texture = face_texture

        sampled_texture = F.grid_sample(face_texture, uv.permute(0, 2, 3, 1),  # uv:(B, H, W, 2)
                                        mode='bilinear',
                                        align_corners=True)  # (B, feature_dim, H, W)

        feat = sampled_texture[:, :self.face_texture_channels, :, :]
        alpha = sampled_texture[:, self.face_texture_channels:, :, :]

        # sampled_texture = sampled_texture * mask
        alpha = torch.sigmoid(alpha)

        feat = F.interpolate(feat, (self.volume_size, self.volume_size), mode='bilinear')
        alpha = F.interpolate(alpha, (self.volume_size, self.volume_size), mode='bilinear')

        mask = (uv != 0).all(1, keepdim=True).float()
        mask = F.interpolate(mask, size=(self.volume_size, self.volume_size), mode='nearest')
        mask = (mask > 0.99).all(1, keepdim=True)

        feat = feat * mask
        alpha = alpha * mask

        return feat, alpha, mask
