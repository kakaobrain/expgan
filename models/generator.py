import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import get_affine


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_dim = cfg.model.z_dim
        self.w_dim = cfg.model.w_dim
        self.tex_dim = cfg.model.texture_dim
        # self.use_aux = cfg.model.use_aux
        self.args_stylegan = cfg.model.stylegan
        self.shape_dim = 103
        self.volume_size = cfg.model.volume_size

        # self.face_texture_name = cfg.model.generator.face_texture
        # self.decoder_name = cfg.model.generator.decoder

        # Face Texture
        # if self.face_texture_name == 'PiGAN':
        #     from .PiGAN.facetexture import FaceTextureDecoder
        #     self.texture_decoder = FaceTextureDecoder(
        #                                 z_dim=self.w_dim,
        #                                 texture_dim=self.tex_dim,
        #                                 out_size=cfg.dataset.image_size,
        #                                 # out_size=self.volume_size,
        #                                 norm_type=cfg.model.textureDecoder.norm_type)
        #     if cfg.model.EG3D.face_feature_channels > 0:
        #         print(f'Change Hyperparameters(cfg.model.EG3D.face_feature_channels) : {cfg.model.EG3D.face_feature_channels} -> {0}')
        #         cfg.model.EG3D.face_feature_channels = 0
        # elif self.face_texture_name == 'EG3D':
        #     assert cfg.model.generator.decoder == 'EG3D'
        #     assert cfg.model.EG3D.face_feature_channels > 0
        # else:
        #     raise KeyError(f'Invalid Face Texture Module : {cfg.model.generator.face_texture}')

        # Decoder
        # if self.decoder_name == 'EG3D':
        from .EG3D.eg3d_decoder import EG3DDecoder
        from .EG3D.mapping_network import MappingNetwork

        # self.args_decoder = cfg.model.EG3D

        self.decoder = EG3DDecoder(
                            w_dim=self.w_dim,
                            feature_dim=self.tex_dim,
                            volume_size=self.volume_size,
                            **cfg.model.EG3D)

        # if self.face_texture_name == 'PiGAN':
        #     num_ws = self.decoder.num_ws + 2
        # elif self.face_texture_name == 'EG3D':
        num_ws = self.decoder.num_ws + 1
        self.mapping_network = MappingNetwork(
                                    z_dim=self.z_dim+self.shape_dim,
                                    c_dim=0,
                                    w_dim=self.w_dim,
                                    num_ws=num_ws
                                )
        self.register_buffer('mean_w', torch.zeros(num_ws, self.w_dim))
        # elif self.decoder_name == 'PiGAN':
        #     from .StyleGAN2.mapping_network import MappingNetwork
        #     print(f'Change Hyperparameters(self.w_dim) : {self.w_dim} -> {self.w_dim}')
        #     self.w_dim *= 2
        #     self.mapping_network = MappingNetwork(
        #                                     input_dim=self.z_dim+self.shape_dim,
        #                                     out_dim=self.w_dim,
        #                                     hidden_dim=self.z_dim
        #                                 )
        #     self.register_buffer('mean_w', torch.zeros(self.w_dim))
        # else:
        #     raise KeyError(f'Invalid Generator Decoder : {cfg.model.generator.decoder}')

        # Renderer
        from .StyleGAN2.renderer import StyleRenderer
        self.renderer = StyleRenderer(
                            z_dim=self.w_dim,
                            tex_dim=self.tex_dim,
                            in_size=self.volume_size,
                            out_size=cfg.dataset.image_size,
                            **self.args_stylegan)

    def get_generator_params_for_optim(self):
        G_param_decoder = list(self.decoder.parameters())
        G_param_stylegan = list(self.renderer.parameters())

        # if self.decoder_name == 'EG3D':
        G_param_decoder += list(self.mapping_network.parameters())
        # if self.face_texture_name == 'PiGAN':
        #     G_param_decoder += list(self.texture_decoder.parameters())

        return G_param_decoder, G_param_stylegan

    def get_latent(self, shape, update_mean=True, truncation=None, z=None, w=None):
        if z is None:
            z = torch.randn(shape.shape[0], self.z_dim, device=shape.device)
        z = torch.cat([z, shape], dim=-1)

        if w is None:
            w = self.mapping_network(z)

        if update_mean:
            self.mean_w = w.mean(0)

        if truncation is not None:
            w = torch.lerp(self.mean_w.unsqueeze(0), w, truncation)

        # if self.decoder_name == 'EG3D':
        num_ws = self.decoder.num_ws
        w_decoder = w[:, :num_ws, :]
        # if self.face_texture_name == 'PiGAN':
        #     w_face_texture = w[:, num_ws:num_ws+1, :]
        #     num_ws += 1
        # else:
        w_face_texture = None
        w_renderer = w[:, num_ws:num_ws+1, :]
        num_ws += 1
        # else:
        #     w_face_texture = w[:, :self.w_dim//2]
        #     w_decoder = w[:, self.w_dim//2:]
        #     w_renderer = w

        return w_face_texture, w_decoder, w_renderer

    def get_face_info(self, w, uv, c2w_tex, depth):
        face_info = dict()

        # if self.face_texture_name == 'PiGAN':
        #     sampled_face_texture, sampled_face_alpha = self.texture_decoder(w, uv, c2w_tex)

        #     resize_factor = sampled_face_texture.shape[-1] // self.volume_size
        #     sampled_face_texture = F.avg_pool2d(sampled_face_texture, kernel_size=resize_factor, stride=resize_factor)
        #     sampled_face_alpha = F.avg_pool2d(sampled_face_alpha, kernel_size=resize_factor, stride=resize_factor)

        #     face_info['face_texture'] = sampled_face_texture
        #     face_info['face_alpha'] = sampled_face_alpha
        # else:
        assert w is None

        depth = F.interpolate(depth, size=(self.volume_size, self.volume_size), mode='nearest')
        face_info['face_depth'] = depth
        face_info['uv'] = uv

        return face_info

    def forward(self, shape, c2w, uv, depth, truncation=None, update_mean=True, c2w_tex=None, z=None, w=None):
        '''
        inputs
            shape: (B, 103)
            uv : (B, 2, H, W)
            c2w : camera to world transform_matrix
            depth : (B, 1, H, W)
        outputs
            mask : (B, 1, H, W)
            pred : (B, 3, H, W)
            mu, sigma : (B, z_dim)
        '''
        # if stage is not None:
        #     out_size = stage['image_size']
        # else:
        #     out_size = None

        if c2w_tex is None:
            c2w_tex = c2w

        # uv = F.interpolate(uv, size=(self.volume_size, self.volume_size), mode='nearest')
        # depth = F.interpolate(depth, size=(self.volume_size, self.volume_size), mode='nearest')

        w_face_texture, w_decoder, w_renderer = self.get_latent(shape, update_mean, truncation, z=z, w=w)

        # depth and uv map for face texture sampling;
        # if use a separate face texture decoder, also contains all face texture and alpha
        face_info = self.get_face_info(w_face_texture, uv, c2w_tex, depth)

        output = self.decoder(w_decoder, c2w, face_info, c2w_face=c2w_tex)

        render_input = output['pred']  # (B, H, W, feature_dim)
        depth_out = output['depth']  # (B, H, W, 1)
        face_alpha = output['face_alpha']
        face_texture = output['face_texture']

        render_input = render_input.permute(0, 3, 1, 2).contiguous()  # (B, feature_dim, H, W)
        depth_out = depth_out.permute(0, 3, 1, 2).contiguous()  # (B, 1, H, W)

        aux = render_input[:, :3]

        # concat and render
        pred = self.renderer(render_input, w_renderer)
        pred = torch.tanh(pred)

        return {
            'pred': pred,
            'depth': depth_out,
            'aux_out': aux,  # affine transformed
            'face_alpha': face_alpha,
            'sampled_face_texture': face_texture,
            'volume_feature': render_input
        }

    # def reparameterize(self, mean, logvar):
    #     std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
    #     epsilon = torch.randn_like(std)
    #     return epsilon * std + mean


if __name__ == '__main__':
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--cfg', type=str)
    args = parser.parse_args()

    batch_size = 12
    volume_size = 512
    stage = {'epoch_end': None, 'image_size': 256, 'lr_mul': 1, 'batch_size': 12}
    # stage = None

    shape = torch.randn((batch_size, 103), dtype=torch.float32).cuda()
    c2w = torch.randn((batch_size, 3, 4), dtype=torch.float32).cuda()
    uv = torch.rand((batch_size, 2, volume_size, volume_size), dtype=torch.float32).cuda()
    depth = 1 + torch.rand((batch_size, 1, volume_size, volume_size), dtype=torch.float32).cuda()
    transform = False
    truncation = None
    update_mean = True
    c2w_tex = None

    cfg = OmegaConf.load(args.cfg)
    net_G = Generator(cfg)
    net_G.cuda()

    out = net_G(shape, c2w, uv, depth, transform, stage, truncation, update_mean, c2w_tex)
