from omegaconf import OmegaConf
import torch
import pickle

# from .decalib.utils.config import cfg as deca_cfg
from .decalib.renderer import DepthRenderer
from .decalib.FLAME import FLAME
from .decalib.utils import util
from pytorch_lightning.core.lightning import LightningModule
import torch.nn.functional as F
import numpy as np


from pytorch3d.io import load_obj
from pytorch3d.transforms import so3_exp_map

def batch_lstsq(
    input: torch.Tensor,  # matrix B of shape (batch * m * k) 
    A: torch.Tensor  # matrix A of shape (batch * m * n) 
):

    # X = torch.bmm(
    #     torch.pinverse(A),
    #     input
    # )

    AT = A.permute(0,2,1)
    
    X = torch.solve(
        torch.bmm(AT, input),
        torch.bmm(AT, A)
    ).solution

    # X = torch.bmm(
    #     torch.inverse(torch.bmm(AT, A)),
    #     torch.bmm(AT, input)
    # )
    
    return X

def batch_bounding_box(batch_landmarks):
    batch_crop = []

    for lm in batch_landmarks:
        lm = lm.detach().cpu().numpy()
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        # qsize = np.hypot(*x) * 2

        # Crop.
        crop = (min(quad[:,0]), min(quad[:,1]), max(quad[:,0]), max(quad[:,1]))
        batch_crop.append(torch.tensor(crop).float().to(batch_landmarks.device))

    return torch.stack(batch_crop, dim=0)


class FlameDecoder(LightningModule):
    '''
    '''
    def __init__(self, image_size, deca_dir, deca_cfg, masking=True):
        #
        super().__init__()

        self.cfg = OmegaConf.merge(OmegaConf.load(deca_cfg), OmegaConf.create({'deca_dir': deca_dir}))
        # self.cfg = deca_cfg
        # if config is None:
        #     self.cfg = deca_cfg
        # else:
        #     self.cfg = config
    
        fn_obj = self.cfg.model.topology_path
        self.cop = self.cfg.dataset.cop
        self.cfg.model.use_tex = False
        
        # load topology
        if masking:
            masking_index = './data/FFHQ/indices_ear_noeye.pkl'
            masking_index = pickle.load(open(self.cfg.masking_index, 'rb'))

        _, faces, aux = load_obj(fn_obj)

        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)[..., :-1]

        faces = faces.verts_idx[None, ...]
        # uvs = aux.verts_uvs[None, ...]

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.
        face_colors = util.face_vertices(colors, faces)

        # face_uvcoords[~indices] = 0

        if masking:
            indices = torch.tensor(masking_index, device=faces.device)
            indices = (faces[..., None,:] == indices[None, :,None]).any(2).all(-1) 

            faces = faces[indices]
            face_uvcoords = face_uvcoords[indices]
            face_colors = face_colors[indices]

        # self.faces = faces
        # self.face_uvcoords = face_uvcoords
        self.register_buffer('face_colors', face_colors)
        self.register_buffer('faces', faces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        
        # define renderer
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
            'cull_backfaces': False
        }
        self.renderer = DepthRenderer(raster_settings)
        
        # load flame model
        self.flame = FLAME(self.cfg.model)
        self.freeze()
        
    def forward(self, codedict, bbox, trans_factor=None, pose=None, realign=True, mesh_render=False):     
        device = bbox.device

        sz = bbox[:,2] - bbox[:,0]
        ctr = (bbox[:,:2] + bbox[:,2:]) * 0.5
            
        opdict = self.decode_ptrs(codedict)
        
        k2d = opdict['landmarks3d'][..., :2].clone()  # (B, n_kpt, 2) 
        # round
        k2d = torch.round(k2d * 1000.) / 1000.
        rvec = codedict['pose'][:, :3] # (B, 3)
        if pose is not None:
            rvec = pose
    
        codedict_canonical = {
            'shape': codedict['shape'],
            'exp': codedict['exp'],
            'pose': codedict['pose'].clone(),
            'cam': codedict['cam'].clone()
        }
        codedict_canonical['pose'][:, :3] *= 0
        codedict_canonical['cam'][:, 0] = 10.0
        codedict_canonical['cam'][:, 1] = 0.0
        codedict_canonical['cam'][:, 2] = 0.0
        opdict = self.decode_ptrs(codedict_canonical)
                
        # k3d = np.around(opdict['landmarks3d'][0].cpu().numpy().astype(float), 3) #* 0.25  
        # vertices = opdict['transformed_vertices'][0].cpu().detach().numpy() #* 0.25

        k3d = opdict['landmarks3d']
        vertices = opdict['transformed_vertices'].to(torch.float32)
        # round 
        k3d = torch.round(k3d * 1000.) / 1000.
        # push back vertices; just to make depth range positive
        k3d[..., 2] -= 1.5
        vertices[..., 2] -= 1.5

        # map 2d landmarks and vertices to the full image        
        k2d = k2d * (sz.view(sz.shape[0], 1, 1) * 0.5)
        k2d[..., 0] += (ctr[...,0:1] - self.cop)
        k2d[..., 1] -= (ctr[...,1:2] - self.cop)
        k2d /= self.cop

        R = so3_exp_map(rvec) # (B, 3, 3)

        k3d_r = torch.bmm(k3d, R.permute(0,2,1))  # (B, 68, 3)
        lhs_x = torch.cat([
            -k2d[...,:1],
            torch.ones_like(k3d_r[...,2:]),
            torch.zeros_like(k3d_r[...,2:]),
        ], dim=-1)

        lhs_y = torch.cat([
            -k2d[...,1:2],
            torch.zeros_like(k3d_r[...,2:]),
            torch.ones_like(k3d_r[...,2:])
        ], dim=-1)
    
        rhs_x = -k3d_r[..., :1]
        rhs_y = -k3d_r[..., 1:2]
        lhs = torch.cat([lhs_x, lhs_y], dim=-2).to(torch.float32)
        rhs = torch.cat([rhs_x, rhs_y], dim=-2).to(torch.float32)

        X = torch.linalg.lstsq(lhs, rhs).solution
        # X = batch_lstsq(rhs, lhs)
        
        iss, tx, ty = X[:,0], X[:,1], X[:,2] 
        ss = 1.0 / iss
        assert (ss >= 0).all(), 'Negative scale is not allowed.'
        
        # translate 
        if trans_factor is not None:
            sign = torch.where(tx < 0, torch.ones_like(tx)*-1, torch.ones_like(tx))
            tx += (trans_factor[...,0:1] * sign)
            ty += trans_factor[...,1:2]

        tvec = torch.cat([tx, ty, torch.zeros_like(tx)], dim=-1).unsqueeze(-1) # (B, 3, 1)


        Rt = torch.cat([R, tvec], dim=-1)
        sRt = ss.unsqueeze(-1) *  Rt
        homo = torch.tensor([[0,0,0,1]], device=device).unsqueeze(0).expand(sRt.shape[0], -1, -1)
        sRt = torch.cat([sRt, homo], dim=1)
        isRt = torch.inverse(sRt)
        
        # render depth and uv
        tv = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        tv = torch.bmm(tv, sRt.permute(0,2,1))
        tv = tv[..., :3]
        # verts[..., -1] *= iss

        if realign or pose is not None:
            t_landmarks = self.flame.seletec_3d68(tv)
            img_size = self.renderer.raster_settings['image_size']
            norm_factors = torch.tensor([2, -2], device=tv.device)
            img_landmarks = (t_landmarks[..., :2] / norm_factors.view(1, 1, -1) + 0.5) * (img_size - 1)
            bounding_box = batch_bounding_box(img_landmarks)
            bounding_box = (bounding_box / (img_size - 1) - 0.5) * norm_factors.repeat(1, 2)

            bb_scale = (bounding_box[:, 2] - bounding_box[:, 0]) * 0.5
            bb_offset = (bounding_box[:, :2] + bounding_box[:, 2:]) * 0.5
            tv[..., :2] -= bb_offset.view(-1, 1, 2)
            tv /= bb_scale.view(-1, 1, 1)

            sRt[:, :2, 3] -= bb_offset
            sRt[:, :3, :4] /= bb_scale.view(-1, 1, 1)
        isRt, isRt_orig = torch.inverse(sRt), isRt

        if mesh_render:
            # mesh img
            _tv = tv.clone()
            _tv[:,:,1:] = -_tv[:,:,1:]
            shape_images = self.render_shape(vertices, _tv)
        else : shape_images = None

        z0 = torch.bmm(sRt, isRt[...,-1].unsqueeze(-1))[:,2] # must be 0.0

        z = z0.view(z0.shape[0], 1, 1) - tv[:, :, 2:].repeat(1, 1, 3).clone()
        # tv[:, :, 2] = tv[:, :, 2] + 10.0

        min_z = torch.min(z, dim=1, keepdim=True).values
        max_z = torch.max(z, dim=1, keepdim=True).values
        # max_z = torch.max(z)
        # range_z = torch.max(z) - torch.min(z)
        range_z = max_z - min_z
        z = (z - min_z) / range_z
        tv[:, :, 2] = -tv[:, :, 2] + 10.0

        # attributes = util.face_vertices(z, faces.expand(1, -1, -1))
        attributes = torch.cat([util.face_vertices(z, self.faces.expand(z.shape[0], -1, -1)),
                               self.face_uvcoords.expand(z.shape[0], -1, -1, -1)], dim=-1)

        out = self.renderer(tv, self.faces.expand(z.shape[0], -1, -1), attributes)
        out = out.to(torch.float32)
        d_img = out[:,:3]
        uv_img = out[:,3:]

        uv_img = torch.flip(uv_img, [2])
        d_img = torch.flip(d_img[:,0:1], [2])
        d_img = d_img * range_z[...,:1].unsqueeze(-1) + min_z[...,:1].unsqueeze(-1)

        return {
            'c2w': isRt[:, :3, :],
            'c2w_orig': isRt_orig[:, :3, :],
            'scale': ss,
            'depth': d_img,
            'uv': uv_img,
            'shape_image': shape_images,
            'landmarks3d': self.flame.seletec_3d68(tv),
        }

    def render_shape(self, vertices, transformed_vertices, images=None, detail_normal_images=None, lights=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()*1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        tv = transformed_vertices.clone().detach()
        tv[:,:,2] = tv[:,:,2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(tv, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_colors.expand(batch_size, -1, -1, -1), 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals], 
                        -1)
        # rasterize
        rendering = self.renderer(tv, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        return shape_images

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)
        
    @torch.no_grad()
    def decode_ptrs(self, codedict):
        # images = codedict['images']
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])

        trans_verts = util.batch_orth_proj(verts, codedict['cam'])

        ## output
        opdict = {
            'vertices': verts,
            'transformed_vertices': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
        }
        return opdict
        

# # ======== prev version
# import torch
# import numpy as np
# import pickle
# import cv2

# from decalib.utils.config import cfg as deca_cfg
# from decalib.renderer import DepthRenderer
# from decalib.FLAME import FLAME
# from decalib.utils import util
# from pytorch3d.io import load_obj

# class FlameDecoder(object):
    
#     def __init__(self, image_size, config=None):
        
#         if config is None:
#             self.cfg = deca_cfg
#         else:
#             self.cfg = config
    
#         fn_obj = self.cfg.model.topology_path
#         self.cop = self.cfg.dataset.cop
#         self.cfg.model.use_tex = False
#         self.cfg.device_id = torch.cuda.current_device()
        
#         # load topology
#         masking_index = './data/FFHQ/indices.pkl'
#         masking_index = pickle.load(open(masking_index, 'rb'))

#         _, faces, aux = load_obj(fn_obj)

#         uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
#         uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
#         uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
#         uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
#         face_uvcoords = util.face_vertices(uvcoords, uvfaces)[..., :-1]

#         faces = faces.verts_idx[None, ...]
#         # uvs = aux.verts_uvs[None, ...]
#         indices = torch.tensor(masking_index, device=faces.device)
#         indices = (faces[..., None,:] == indices[None, :,None]).any(2).all(-1) 

#         #face_uvcoords = face_uvcoords[indices]
#         #faces = faces[indices]
#         face_uvcoords[~indices] = 0
        
#         self.faces = faces
#         self.face_uvcoords = face_uvcoords
        
#         # define renderer
#         raster_settings = {
#             'image_size': image_size,
#             'blur_radius': 0.0,
#             'faces_per_pixel': 1,
#             'bin_size': None,
#             'max_faces_per_bin': None,
#             'perspective_correct': False,
#             'cull_backfaces': False
#         }
#         self.renderer = DepthRenderer(raster_settings)
        
#         # load flame model
#         self.flame = FLAME(self.cfg.model)
        
#     def render(self, codedict, bbox, trans_factor=None):
#         bbox = np.array(bbox)

#         sz = bbox[2] - bbox[0]
#         ctr = (bbox[:2] + bbox[2:]) * 0.5
            
#         opdict = self.decode_ptrs(codedict)
        
#         k2d = np.around(opdict['landmarks3d'][0].cpu().numpy().astype(float), 3)[:, :2]
#         k2d = k2d[:, :2].copy()
#         print('k2d', k2d)
#         rvec = np.reshape(codedict['pose'].detach().cpu().numpy()[0, :3], (3, 1)).copy()

#         codedict['pose'][:, :3] *= 0
#         codedict['cam'][:, 0] = 10.0
#         codedict['cam'][:, 1] = 0.0
#         codedict['cam'][:, 2] = 0.0
#         opdict = self.decode_ptrs(codedict)
                
#         k3d = np.around(opdict['landmarks3d'][0].cpu().numpy().astype(float), 3) #* 0.25 
#         vertices = opdict['transformed_vertices'][0].cpu().detach().numpy() #* 0.25
#         # push back vertices; just to make depth range positive
#         k3d[:, 2] -= 1.5
#         vertices[:, 2] -= 1.5

#         # map 2d landmarks and vertices to the full image
#         k2d *= sz * 0.5
#         k2d[:, 0] += (ctr[0] - self.cop)
#         k2d[:, 1] -= (ctr[1] - self.cop)
#         k2d /= self.cop

#         R = cv2.Rodrigues(rvec)[0]
#         print('R', R)

#         k3d_r = k3d @ R.T # (N, 3)
#         lhs_x = np.hstack([
#             -k2d[:, :1], 
#             np.ones_like(k3d_r[:, 2:]),
#             np.zeros_like(k3d_r[:, 2:])
#         ])
#         lhs_y = np.hstack([
#             -k2d[:, 1:2], 
#             np.zeros_like(k3d_r[:, 2:]),
#             np.ones_like(k3d_r[:, 2:])
#         ])
#         rhs_x = -k3d_r[:, :1]
#         rhs_y = -k3d_r[:, 1:2]
#         lhs = np.vstack([lhs_x, lhs_y])
#         rhs = np.vstack([rhs_x, rhs_y])
#         X = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
#         print('X', X)

#         iss, tx, ty = X.ravel()
#         ss = 1.0 / iss
#         assert ss >= 0, 'Negative scale is not allowed.'
        
#         # translate 
#         if trans_factor is not None:
#             sign = -1 if tx < 0 else 1
#             tx = trans_factor[0]
#             tx *= sign
#             ty += trans_factor[1]
        
#         tvec = np.reshape(np.array([tx, ty, 0.0]), (3, 1))
        
#         Rt = np.hstack([R, tvec])
#         sRt = ss * Rt
#         sRt = np.vstack([sRt, np.array([[0, 0, 0, 1]])])
#         isRt = np.linalg.inv(sRt)
        
#         # render depth and uv
#         verts = np.hstack([vertices, np.ones_like(vertices[:, :1])])
#         verts = verts @ sRt.T
#         verts = verts[:, :3]
#         # verts[:, -1] *= iss

#         z0 = (sRt @ isRt[:, -1])[2] # must be 0.0
#         print(z0)

#         tv = torch.from_numpy(verts).unsqueeze(0)
#         z = z0 - tv[:, :, 2:].repeat(1, 1, 3).clone()
#         # tv[:, :, 2] = tv[:, :, 2] + 10.0

#         print(z.shape)

#         min_z = torch.min(z)
#         # max_z = torch.max(z)
#         range_z = torch.max(z) - torch.min(z)
#         z = (z - min_z) / range_z
#         print('max z', torch.max(z))
#         tv[:, :, 2] = -tv[:, :, 2] + 10.0

#         # attributes = util.face_vertices(z, faces.expand(1, -1, -1))
#         attributes = torch.cat([util.face_vertices(z, self.faces.expand(1, -1, -1)),
#                                self.face_uvcoords], dim=-1)

#         device_id = torch.cuda.current_device()
#         out = self.renderer(tv.to(device_id), self.faces.expand(1, -1, -1).to(device_id), attributes.to(device_id))
#         out = out.to(torch.float32)
#         d_img = out[:,:3]
#         uv_img = out[:,3:]
        
#         uv_img = torch.flip(uv_img[0], [1])
#         d_img = torch.flip(d_img[0,0:1], [1])
#         d_img = d_img * range_z + min_z
        
#         # # d_img = renderer(tv.to('cuda'), faces.expand(1, -1, -1).to('cuda'), attributes.to('cuda'))
#         # d_img = np.transpose(d_img.detach().cpu().numpy()[0], (1, 2, 0))[:, :, 0]
#         # d_img = d_img * range_z.numpy() + min_z.numpy()
        
#         return {
#             'c2w': isRt[:3, :].tolist(),
#             'scale': ss,
#             'depth': d_img,
#             'uv': uv_img,
#         }
        
#     @torch.no_grad()
#     def decode_ptrs(self, codedict):
#         # images = codedict['images']
#         ## decode
#         verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
#         ## projection
#         landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
#         landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])

#         trans_verts = util.batch_orth_proj(verts, codedict['cam'])

#         ## output
#         opdict = {
#             'vertices': verts,
#             'transformed_vertices': trans_verts,
#             'landmarks2d': landmarks2d,
#             'landmarks3d': landmarks3d,
#         }
#         return opdict