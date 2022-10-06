import numpy as np
import torch
import torch.nn as nn

from .face_texture import FaceTexture
from .networks_stylegan2 import SynthesisBlock
from .triplane_decoder import TriplaneDecoder


class EG3DRendererBase(nn.Module):
    '''
    [Inputs]
        - w => (B, w_dim)
        - points => (B, V * V, #step, 3)
    [Outputs]
        - output: (B, V * V, #step, out_feature_channels + 1)
    '''
    def __init__(
        self,
        w_dim,
        img_feature_channels,
        face_feature_channels,
        volume_size,
        triplane_resolution,
        triplane_channels,
        triplane_decoder_dim,
        triplane_decoder_layers,
    ):
        super().__init__()

        self.w_dim = w_dim
        self.img_feature_channels = img_feature_channels
        self.face_feature_channels = face_feature_channels
        self.triplane_resolution = triplane_resolution
        self.triplane_channels = triplane_channels

        self.num_ws = 0

        self.triplane_decoder = TriplaneDecoder(
            self.triplane_channels,
            self.img_feature_channels,
            hidden_channels=triplane_decoder_dim,
            layers=triplane_decoder_layers,
        )

    def forward(self, ws, points, face_info, update_emas=None):
        raise NotImplementedError

    def build_blocks(self, w_dim, resolutions, input_channels=0, output_channels=3, attr_format='b{}', channel_base=32768, channel_max=512):
        channels_list = []
        channels_list += [input_channels]
        channels_list += [min(channel_base // res, channel_max) for res in resolutions[:-1]]
        channels_list += [output_channels]

        for i, (in_channels, out_channels, resolution) in enumerate(zip(channels_list[:-1], channels_list[1:], resolutions)):
            use_fp16 = False  # (res >= fp16_resolution)
            is_last = (i == len(resolutions) - 1)
            block = SynthesisBlock(
                        in_channels, out_channels,
                        w_dim=w_dim,
                        resolution=resolution,
                        img_channels=output_channels,
                        is_last=is_last,
                        use_fp16=use_fp16,
                        use_noise=False
                    )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, attr_format.format(i), block)

    def to_block_ws(self, ws, resolutions, attr_format='b{}', w_idx_offset=0):
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = w_idx_offset
        for i in range(len(resolutions)):
            block = getattr(self, attr_format.format(i))
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv
        return block_ws, w_idx + block.num_torgb

    def forward_blocks(self, block_ws, x=None, img=None, attr_format='b{}', update_emas=False):
        for i, cur_ws in enumerate(block_ws):
            block = getattr(self, attr_format.format(i))
            x, img = block(x, img, cur_ws, update_emas=update_emas, noise_mode='none')
        return img


class EG3DRenderer(EG3DRendererBase):
    def __init__(
        self,
        w_dim,
        img_feature_channels,
        face_feature_channels,
        volume_size,
        triplane_resolution,
        triplane_channels,
        triplane_decoder_dim,
        triplane_decoder_layers,
    ):
        super().__init__(
            w_dim,
            img_feature_channels,
            face_feature_channels,
            volume_size,
            triplane_resolution,
            triplane_channels,
            triplane_decoder_dim,
            triplane_decoder_layers,
        )

        self.triplane_channel_idx = 3 * self.triplane_channels

        out_channels = self.triplane_channel_idx
        if self.face_feature_channels > 0:
            out_channels += self.face_feature_channels
            # out_channels += (self.face_feature_channels + 1)
            self.face_texture = FaceTexture(face_feature_channels, img_feature_channels, volume_size)

        self.triplane_block_attr_format = 'triplane_block_{}'

        self.triplane_block_resolutions = [2 ** i for i in range(2, int(np.log2(self.triplane_resolution)) + 1)]
        self.build_blocks(
            self.w_dim,
            self.triplane_block_resolutions,
            output_channels=out_channels,
            attr_format=self.triplane_block_attr_format
        )

    def forward(self, ws, face_info, update_emas=None, c2w=None):
        if update_emas is None:
            update_emas = self.training

        triplane_block_ws, w_idx_offset = self.to_block_ws(ws, self.triplane_block_resolutions, attr_format=self.triplane_block_attr_format)

        coarse_output = self.forward_blocks(triplane_block_ws, attr_format=self.triplane_block_attr_format, update_emas=update_emas)
        triplane = coarse_output[:, :self.triplane_channel_idx, :, :].contiguous()

        if hasattr(self, 'face_texture'):
            # EG3D Face Texture
            uv = face_info['uv']
            face_texture_plane = coarse_output[:, self.triplane_channel_idx:, :, :].contiguous()
            face_texture, face_alpha, face_mask = self.face_texture(face_texture_plane, uv, c2w=c2w)

            face_info['face_texture'] = face_texture
            face_info['face_alpha'] = face_alpha

        return triplane, face_info


class EG3DRendererSplit(EG3DRendererBase):
    def __init__(
        self,
        w_dim,
        img_feature_channels,
        face_feature_channels,
        volume_size,
        triplane_resolution,
        triplane_channels,
        triplane_decoder_dim,
        triplane_decoder_layers,
    ):
        super().__init__(
            w_dim,
            img_feature_channels,
            face_feature_channels,
            volume_size,
            triplane_resolution,
            triplane_channels,
            triplane_decoder_dim,
            triplane_decoder_layers,
        )

        self.triplane_block_attr_format = 'triplane_block_{}'
        self.triplane_block_resolutions = [2 ** i for i in range(2, int(np.log2(self.triplane_resolution)) + 1)]
        self.build_blocks(
            self.w_dim,
            self.triplane_block_resolutions,
            output_channels=3*self.triplane_channels,
            attr_format=self.triplane_block_attr_format
        )

        if self.face_feature_channels > 0:
            self.face_texture_block_attr_format = 'face_texture_block_{}'
            self.face_texture_block_resolutions = [2 ** i for i in range(2, int(np.log2(self.triplane_resolution)) + 1)]
            self.build_blocks(
                self.w_dim,
                self.triplane_block_resolutions,
                output_channels=self.face_feature_channels,
                attr_format=self.face_texture_block_attr_format
            )

            self.face_texture = FaceTexture(face_feature_channels, img_feature_channels, volume_size)

    def forward(self, ws, face_info, update_emas=None, c2w=None):
        if update_emas is None:
            update_emas = self.training

        triplane_block_ws, w_idx_offset = self.to_block_ws(ws, self.triplane_block_resolutions, attr_format=self.triplane_block_attr_format)
        triplane = self.forward_blocks(triplane_block_ws, attr_format=self.triplane_block_attr_format, update_emas=update_emas)

        # EG3D Face Texture
        if hasattr(self, 'face_texture'):
            face_texture_block_ws, w_idx_offset = self.to_block_ws(ws, self.triplane_block_resolutions, attr_format=self.face_texture_block_attr_format)
            face_texture_plane = self.forward_blocks(face_texture_block_ws, attr_format=self.face_texture_block_attr_format, update_emas=update_emas)

            uv = face_info['uv']
            face_texture, face_alpha, face_mask = self.face_texture(face_texture_plane, uv, c2w=c2w)

            face_info['face_texture'] = face_texture
            face_info['face_alpha'] = face_alpha

        return triplane, face_info
