# a class for evaluating model 
import torch.nn as nn
import torch

import torch.nn.functional as F
from models.flamedecoder import FlameDecoder

from models.generator import Generator

class EXPGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net_G = Generator(cfg)
        self.net_G_ema = Generator(cfg)
        self.flamedecoder = FlameDecoder(**cfg.model.flamedecoder)
        self.cfg = cfg
    
    def load_from_checkpoint(self, fn_ckpt):
        ckpt = torch.load(fn_ckpt)['state_dict']
        _, _ = self.load_state_dict(ckpt, strict=False)

    @torch.no_grad()
    def forward(self, shape, c2w, uv, depth, tform_param=None, ema=True): 
        if ema:
            output = self.net_G_ema(shape, c2w, uv, depth, tform_param)
        else: 
            output = self.net_G(shape, c2w, uv, depth, tform_param)

        return output