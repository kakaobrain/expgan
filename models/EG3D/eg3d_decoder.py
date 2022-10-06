from tokenize import Triple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..volume_rendering import (get_ray_volume_ortho, render_volume,
                                transform_ray, sample_pdf, sort_by_depth)


class EG3DDecoder(nn.Module):
    def __init__(
        self,
        w_dim,
        feature_dim,
        volume_size,
        coarse_steps,
        fine_steps,
        hierarchical_sample,
        fov,
        ray_start,
        ray_end,
        perturb,
        last_back,
        triplane_resolution,
        triplane_decoder_dim,
        triplane_decoder_layers,
        face_feature_channels=None,
        face_feature_split=False,
    ):
        super().__init__()

        self.w_dim = w_dim
        self.feature_dim = feature_dim
        self.volume_size = volume_size
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.hierarchical_sample = hierarchical_sample
        self.fov = fov
        self.ray_start = ray_start
        self.ray_end = ray_end
        self.perturb = perturb
        self.last_back = last_back

        if face_feature_split:
            from .eg3d_renderer import EG3DRendererSplit as EG3DRenderer
        else:
            from .eg3d_renderer import EG3DRenderer as EG3DRenderer

        self.renderer = EG3DRenderer(
                            w_dim=w_dim,
                            img_feature_channels=feature_dim,
                            face_feature_channels=face_feature_channels,
                            volume_size=volume_size,
                            triplane_resolution=triplane_resolution,
                            triplane_channels=feature_dim,
                            triplane_decoder_dim=triplane_decoder_dim,
                            triplane_decoder_layers=triplane_decoder_layers,
                        )
        self.step = 0
        self.num_ws = self.renderer.num_ws

    def forward(self, w, c2w, face_info, c2w_face=None):
        '''
        inputs
            w: latent code (B, w_dim)
            c2w: camera transform matrix for ray (B, 3, 4)
            face_info : [sampled face texture, sampled face alpha, gt face depth]
        outputs
            pred: (B, H, W, feature_dim)
            depth : (B, H, W, 1)
        '''
        B, device = w.shape[0], w.device
        if c2w_face is None:
            c2w_face = c2w

        if self.training:
            noise_std = max(0, 1. - self.step/5000.)
        else:
            noise_std = 0.0

        # get ray volume (orthogonal ray)
        points, directions, z_vals, origins = get_ray_volume_ortho(B, self.volume_size, self.volume_size, self.fov, self.ray_start, self.ray_end, self.coarse_steps, device, perturb=self.perturb)

        # transform ray and directions with transform matrix
        tformed_points, tformed_directions, tformed_ray_origins = transform_ray(points, directions, c2w, origins)
        tformed_points[:, :, 0] /= 2.5
        tformed_points[:, :, 1] /= 2.0
        tformed_points[:, :, 2] = (tformed_points[:, :, 2] + 2.) / 2.5
        tformed_points = tformed_points.view(B, self.volume_size * self.volume_size, self.coarse_steps, -1)

        # inference rgb feature and sigma
        triplane, face_info = self.renderer(w, face_info=face_info, c2w=c2w_face)
        coarse_output = self.renderer.triplane_decoder(triplane, tformed_points)

        features = coarse_output[..., :self.feature_dim]
        sigmas = coarse_output[..., self.feature_dim:]

        features = features.view(B, self.volume_size, self.volume_size, -1, self.feature_dim)
        sigmas = sigmas.view(B, self.volume_size, self.volume_size, -1, 1)
        # all_w_vals = w_vals

        if self.hierarchical_sample:
            with torch.no_grad():
                _, _, weights, _ = render_volume(features, sigmas, z_vals, device, noise_std, last_back=False)
                weights = weights.view(B * self.volume_size * self.volume_size, self.coarse_steps) + 1e-5

                #### start new importance sampling
                z_vals = z_vals.reshape(B * self.volume_size * self.volume_size, self.coarse_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.view(B, self.volume_size * self.volume_size, self.coarse_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 self.fine_steps, det=(not self.perturb)).detach()
                fine_z_vals = fine_z_vals.view(B, self.volume_size * self.volume_size, self.fine_steps, 1)

                fine_points = tformed_ray_origins.unsqueeze(2) + tformed_directions.unsqueeze(1) * fine_z_vals
                fine_points[..., 0] /= 2.5
                fine_points[..., 1] /= 2.0
                fine_points[..., 2] = (fine_points[..., 2] + 2.) / 2.5
                #### end new importance sampling

            # Model prediction on re-sampled fine points
            fine_output = self.renderer.triplane_decoder(triplane, fine_points)

            # combine coarse and fine samples
            fine_features = fine_output[..., :self.feature_dim]
            fine_sigmas = fine_output[..., self.feature_dim:]
            fine_features = fine_features.view(B, self.volume_size, self.volume_size, -1, self.feature_dim)
            fine_sigmas = fine_sigmas.view(B, self.volume_size, self.volume_size, -1, 1)

            all_features = torch.cat([fine_features, features], dim=-2)
            all_sigmas = torch.cat([fine_sigmas, sigmas], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            all_z_vals = all_z_vals.view(B, self.volume_size, self.volume_size, -1, 1)
            
            features, sigmas, z_vals = sort_by_depth(all_features, all_sigmas, all_z_vals)

        # concat with face features
        face_texture = face_info['face_texture']
        face_alpha = face_info['face_alpha']
        face_depth = face_info['face_depth']

        assert face_texture.shape[-1] == self.volume_size
        assert face_alpha.shape[-1] == self.volume_size
        assert face_depth.shape[-1] == self.volume_size

        unsqueezed_face_texture = face_texture.permute(0, 2, 3, 1).unsqueeze(-2)  # (B, H, W, 1, feature_dims)
        unsqueezed_face_alpha = face_alpha.permute(0, 2, 3, 1).unsqueeze(-2)  # (B, H, W, 1, 1)
        unsqueezed_face_depth = face_depth.permute(0, 2, 3, 1).unsqueeze(-2)  # (B, H, W, 1, 1)

        unsqueezed_face_info = [unsqueezed_face_texture, unsqueezed_face_alpha, unsqueezed_face_depth]
        # volume rendering
        output, depth, weights, _ = render_volume(features, sigmas, z_vals, device, noise_std, last_back=self.last_back, face_info=unsqueezed_face_info)

        return {
            'pred': output,
            'depth': depth,
            'face_alpha': face_alpha,
            'face_texture': face_texture,
        }

    def upsample(self, tensors: list, size):
        channels = [t.shape[-1] for t in tensors]
        tensors = torch.cat(tensors, dim=-1)
        B, H, W, num_steps, C = tensors.shape
        tensors = tensors.view(B, H, W, -1).permute(0, 3, 1, 2)
        tensors = F.interpolate(tensors, size=size, mode='bilinear', align_corners=True)
        tensors = tensors.permute(0, 2, 3, 1).view(B, size[0], size[1], num_steps, C)

        out_tensors = []
        for ch in channels:
            out_tensors.append(tensors[..., :ch])
            tensors = tensors[..., ch:]

        return out_tensors
