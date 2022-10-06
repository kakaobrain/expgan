import json
import os
import pickle

# import cv2
import numpy as np
import torch
import torchvision.transforms as T
from numpy.random import default_rng
from PIL import Image
from torch.utils.data import Dataset

from .utils import *

rng = default_rng()

class FFHQDataset(Dataset) : 
    def __init__(self, data_dir, fn_meta, fn_meta_flip='', split='train', image_size=512, no_repeat=False) : 
        
        assert split in ['train', 'test', 'val', 'all']

        self.root_dir = data_dir
        self.split = split
        self.img_wh = (image_size, image_size)
        self.fn_meta = fn_meta

        def _load_meta(fn):
            ext_meta = os.path.splitext(fn)[-1]
            if ext_meta == '.json':
                with open(os.path.join(self.root_dir, fn), 'r') as f : 
                    data = json.load(f)
            elif ext_meta == '.pkl':
                data = pickle.load(open(os.path.join(self.root_dir, fn), 'rb'))
            return data['frames']

        self.meta_frames = _load_meta(fn_meta)

        if fn_meta_flip and split == 'train':
            self.meta_frames_flip = _load_meta(fn_meta_flip)
        else:
            self.meta_frames_flip = {}
        
        # self.focal = data['focal']
        
        self.img_dir = os.path.join(self.root_dir, 'img')
        
        self.files = [k for k in self.meta_frames]
        
        if no_repeat:
            self.files = np.unique(self.files)
    
        self._validate_file_list()

    def _validate_file_list(self):
        skip_ids = []
        for fid in self.files:
            if fid not in self.meta_frames:
                skip_ids.append(fid)
                continue
            # for d, e in zip(dirs, exts):
            #     if not os.path.isfile(os.path.join(d, f'{fid}{e}')):
            #         skip_ids.append(fid)
            #         continue
        skip_ids = set(skip_ids)
        files = [f for f in self.files if f not in skip_ids]
        self.files = files
    
    def __len__(self) : 
        return len(self.files)

    def get_pose(self, frame):
        c2w = frame['transform_matrix']
        c2w = np.array(c2w)[:3, :4]
        s = 1 / frame['scale']
        R = c2w[:3,:3] / s
        t = c2w[:3,3]
        euler = rotationMatrixToEulerAngles(R)
        pose = np.concatenate([euler, [s], t[:2]])
        pose = torch.FloatTensor(pose)
        return pose

    def _maybe_flipped_frame(self, image_id):
        # handle flipped images
        if rng.uniform() < 0.5 and image_id in self.meta_frames_flip:
            frame = self.meta_frames_flip[image_id]
            flipped = True
        else:
            frame = self.meta_frames[image_id]
            flipped = False
        return frame, flipped
    
    def __getitem__(self, idx):
        image_id = self.files[idx]

        frame, flipped = self._maybe_flipped_frame(image_id)
        assert frame['file_path'] == image_id

        fn_img = os.path.join(self.img_dir, image_id + '.png')
        
        # real sample
        image = Image.open(fn_img)
        image = image.resize(self.img_wh, Image.BILINEAR)
        image = T.ToTensor()(image)
        image = image * 2.0 - 1.0 # normalize to (-1~ 1)
        if flipped:
            image = T.functional.hflip(image)

        codedict_real = {
            k: torch.FloatTensor(frame[k]) for k in ['pose', 'shape', 'exp', 'cam']
        }
        c2w_real = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
        bbox_real = torch.FloatTensor(frame['bbox'])

        shape_real = torch.cat([codedict_real['shape'], codedict_real['pose'][3:]], dim=0)

        # fake sample 
        rand_idx = self.files[np.random.randint(0, len(self))]
        fake0, _ = self._maybe_flipped_frame(rand_idx)
        # fake0 = self.meta_frames[rand_idx]
        codedict_fake = {
            k: torch.FloatTensor(fake0[k]) for k in ['pose', 'cam']
        }
        bbox_fake = torch.FloatTensor(fake0['bbox'])

        # for random expression and shape
        rand_idx = self.files[np.random.randint(0, len(self))]
        fake1, _ = self._maybe_flipped_frame(rand_idx)
        codedict_fake['exp'] = torch.FloatTensor(fake1['exp'])
        rand_idx = self.files[np.random.randint(0, len(self))]
        fake2, _ = self._maybe_flipped_frame(rand_idx)
        codedict_fake['shape'] = torch.FloatTensor(fake2['shape'])

        ret = {
            'image_id': image_id,
            'image': image,
            'shape_real': shape_real,
            'codedict_fake': codedict_fake,
            'codedict_real': codedict_real,
            'bbox_fake': bbox_fake,
            'bbox_real': bbox_real,
            'c2w_real': c2w_real,
        }
        
        return ret