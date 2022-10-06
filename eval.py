import argparse
import sys, os
import cv2

import torch
from torch.utils.data import DataLoader
import numpy as np

from omegaconf import OmegaConf

from models.expgan import EXPGAN
from dataset.ffhq import FFHQDataset

from tqdm import tqdm
import matplotlib.pyplot as plt


def save_img(pred, path, show=False):
    pred = pred * 0.5 + 0.5
    pred = pred.permute(1,2,0).detach().cpu().numpy()

    if show:
        plt.imshow(pred);plt.show()
        
    pred = (pred * 255).round().astype(np.uint8)[:,:,::-1]
    cv2.imwrite(path, pred)


def main(args):
    fn_cfg = args.cfg
    fn_ckpt = args.ckpt
    batch_size = args.batch_size
    device = args.device
    split = args.split
    num_workers = args.num_workers
    save_root = args.savedir
    psi = args.psi
    n_imgs = args.n_imgs

    cfg = OmegaConf.load(fn_cfg)
    cfg.dataset.fn_meta_flip = None

    cfg.model.EG3D.coarse_steps = args.coarse_steps
    cfg.model.EG3D.fine_steps = args.fine_steps

    model = EXPGAN(cfg)
    model = model.to(device)
    model.eval()
    model.load_from_checkpoint(fn_ckpt)
    
    model.net_G_ema.decoder.perturb = False

    dataset = FFHQDataset(**cfg.dataset, split=split)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    n_batchs = n_imgs // batch_size

    os.makedirs(args.savedir, exist_ok=True)

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dataloader)):
            if bidx >= n_batchs: break 

            seed = torch.seed()
            torch.manual_seed(seed)

            # infer real 
            batch.update({name: tensor.cuda() for name, tensor in batch.items() if type(tensor) == torch.Tensor})
            batch['codedict_real'].update({name: tensor.cuda() for name, tensor in batch['codedict_real'].items() if type(tensor) == torch.Tensor})

            flame_real = model.flamedecoder(batch['codedict_real'], batch['bbox_real'])
            uv, depth, c2w = flame_real['uv'], flame_real['depth'], flame_real['c2w']
            shape = batch['shape_real']

            output = model.net_G_ema(shape, c2w, uv, depth, truncation=psi, update_mean=False)
            for i in range(batch_size):
                pred, img_id = output['pred'][i], batch['image_id'][i]
                save_img(pred, os.path.join(save_root, img_id + '.png'))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--n_imgs', type=int, default=70000)
    parser.add_argument('--psi', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--coarse_steps', type=int ,default=48)
    parser.add_argument('--fine_steps', type=int ,default=48)

    main(parser.parse_args())
