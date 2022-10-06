import torch
from kornia import create_meshgrid
import numpy as np

def get_ray_volume(B, H, W, fov, ray_start, ray_end, num_steps, device, perturb=True) :
    '''
    outputs
        points: (B, H, W, num_steps, 3)
        directions: (B, H, W, 3)
        z_vals: (B, H, W, num_steps, 1)
    '''
    focal = 1/np.tan((2 * np.pi * np.radians(fov) / 360)/2)
    
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]
    i, j = grid.unbind(-1)
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], dim=-1)
    
    # normalize directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).view(1,1,num_steps,1).expand(H,W,-1,-1)
    points = directions.unsqueeze(2) * z_vals
    
    # batchify
    points = points.expand(B, -1, -1, -1, -1)
    z_vals = z_vals.expand(B, -1, -1, -1, -1)
    directions = directions.expand(B, -1, -1, -1)
    
    if perturb:
        points, z_vals = perturb_points(points, z_vals, directions, device)
    
    return points, directions, z_vals

def get_ray_volume_ortho(B, H, W, focal, ray_start, ray_end, num_steps, device, perturb=True):
    '''
    outputs
        points: (B, H, W, num_steps, 3)
        directions: (B, 1, 1, 3)
        z_vals: (B, H, W, num_steps, 1)
        ray origin: (B, H, W, 3)
    '''
    grid = create_meshgrid(H, W, normalized_coordinates=True, device=device)[0]
    i, j = grid.unbind(-1)
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).view(1,1,num_steps,1).expand(H,W,-1,-1)
    directions = torch.tensor([0.0,0.0,-1.0],device=device, dtype=torch.float32).view(1,1,-1).expand(1, 1, -1)

    origins = \
        torch.stack([(i*(focal)), -(j*(focal)), torch.zeros_like(i)], dim=-1)
    points = origins.unsqueeze(2) + directions.unsqueeze(2) * z_vals
    
    # batchify
    points = points.expand(B, -1, -1, -1, -1)
    z_vals = z_vals.expand(B, -1, -1, -1, -1)
    directions = directions.expand(B, -1, -1, -1)
    origins = origins.expand(B, -1, -1, -1)
    
    if perturb:
        points, z_vals = perturb_points(points, z_vals, directions, device)
    
    return points, directions, z_vals, origins

def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[...,1:2,:] - z_vals[...,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset
    
    points = points + offset * ray_directions.unsqueeze(3)
    return points, z_vals

def render_volume(rgb, sigma, z_vals, device, noise_std=0.5, last_back=False, face_info=None): 
    '''
    volume rendering equation
    inputs
        rgb: (B, H, W, num_steps, feature_dim)
        sigma: (B, H, W, num_steps, 1)
        z_vals: (B, H, W, num_steps, 1)
    output
        rgb_final: (B, H, W, feature_dim)
    '''

    deltas = z_vals[...,1:,:] - z_vals[...,:-1,:]
    delta_inf = 1e10 * torch.ones_like(deltas[...,:1,:]) # the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -2) # (B, H, W, num_steps, 1)
    
    noise = torch.randn(sigma.shape, device=device) * noise_std
    
    alphas = 1 - torch.exp(-deltas * torch.relu(sigma + noise))
    
    if face_info is not None:
        face_feature, face_alpha, face_depth = face_info
        rgb = torch.cat([rgb, face_feature], dim=-2)
        alphas = torch.cat([alphas, face_alpha], dim=-2)
        z_vals = torch.cat([z_vals, face_depth], dim=-2)

        rgb = torch.cat([torch.tanh(rgb[...,:3]), rgb[...,3:]], dim=-1) # for aux

        rgb, alphas, z_vals = sort_by_depth(rgb, alphas, z_vals)

    
    alphas_shifted = torch.cat([torch.ones_like(alphas[...,:1,:]), 1-alphas + 1e-10], -2) # # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted, -2)[...,:-1,:]
    weights_sum = weights.sum(-2)
    
    if last_back:
        weights[..., -1,:] += (1 - weights_sum)
    
    rgb_final = torch.sum(weights * rgb, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    
    return rgb_final, depth_final, weights, z_vals

def transform_ray(points, directions, c2w, origins=None) :
    '''
    inputs
        points : (B, H, W, num_steps, 3)
        directions : (B, H, W, 3)
        c2w : (B, 3, 4), camera to world transform matrix
        origins: (B, H, W, 1)
        
    outputs
        tformed_points : (B, H*W*num_steps, 3)
        tformed_directions : (B, num_steps, 3)
        tformed_ray_origins : (B, H*W, 3), for importance sampling
    '''
    B, H, W, num_steps, _ = points.shape
    
    points = torch.cat([points, torch.ones((B, H, W, num_steps, 1), device=points.device)], dim=-1) # homogenous coordinate

    tformed_points = points.view(B, -1, 4)
    tformed_points = torch.bmm(c2w, tformed_points.permute(0,2,1)).permute(0,2,1)
    
    tformed_directions = directions.unsqueeze(-2).expand(-1, -1, -1, num_steps, -1)
    tformed_directions = tformed_directions.reshape(B, -1, 3)
    tformed_directions = torch.bmm(c2w[...,:3], tformed_directions.permute(0,2,1)).permute(0,2,1)
    
    if origins != None: # orthogonal ray
        origins_homo = torch.cat([origins.view(B, -1, 3), torch.ones((B, H*W, 1), device=origins.device)], dim=2)
        tformed_ray_origins = torch.bmm(c2w, origins_homo.permute(0,2,1)).permute(0,2,1)
    else: # perspective ray
        tformed_ray_origins = c2w[...,3].unsqueeze(2).expand(-1, -1, H*W).permute(0,2,1)    
    
    # normalize
    # tformed_directions = tformed_directions / torch.norm(tformed_directions, dim=-1, keepdim=True)

    return tformed_points, tformed_directions, tformed_ray_origins

def sort_by_depth(features, sigmas, depths):
    '''
    inputs : 
        # ray steps + mesh
        rgb: (B, H, W, num_steps+1, feature_dim)  
        sigma: (B, H, W, num_steps+1, 1)
        z_vals: (B, H, W, num_steps+1, 1)
    outputs : 
        rgb: (B, H, W, num_steps+1, feature_dim)
        sigma: (B, H, W, num_steps+1, 1)
        z_vals: (B, H, W, num_steps+1, 1)
    '''
    _, idx = torch.sort(depths, dim=-2)
    sorted_features = torch.gather(features, -2, idx.expand(features.shape))
    sorted_sigmas = torch.gather(sigmas, -2, idx)
    sorted_depths = torch.gather(depths, -2, idx)

    return sorted_features, sorted_sigmas, sorted_depths

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples