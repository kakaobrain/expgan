from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes

class DepthRenderer(object):
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, vertices, faces, attributes):
        '''
        '''
        # mesh to render
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            **self.raster_settings
        )
    #         image_size=raster_settings.image_size,
    #         blur_radius=raster_settings.blur_radius,
    #         faces_per_pixel=raster_settings.faces_per_pixel,
    #         bin_size=raster_settings.bin_size,
    #         max_faces_per_bin=raster_settings.max_faces_per_bin,
    #         perspective_correct=raster_settings.perspective_correct,
    #     )
        vismask = (pix_to_face > -1).float()
        
        # attributes: depth, uv, ...
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
    #     pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals