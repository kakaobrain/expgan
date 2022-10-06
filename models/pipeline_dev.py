import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.transforms import matrix_to_quaternion
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, get_worker_info
from torchvision.utils import make_grid

from models.flamedecoder import FlameDecoder
from models.generator import Generator

from .metric import d_logistic_loss, d_r1_loss, g_nonsaturating_loss


# class for training model with orthographic projection
class Pipeline(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        # loss weights
        self.adv_weight = cfg.train.loss.adv_weight
        self.aux_weight = cfg.train.loss.aux_weight
        self.r1_weight = cfg.train.loss.r1_weight
        self.r1_weight_stylegan = cfg.train.loss.r1_weight_stylegan  # 10
        self.pose_weight = cfg.train.loss.pose_weight
        self.auxgan_weight = cfg.train.auxgan_weight
        self.exp_weight = cfg.train.loss.exp_weight
        self.shape_weight = cfg.train.loss.shape_weight
        self.alpha_weight = cfg.train.loss.alpha_weight

        # Generator
        self.net_G = Generator(cfg)
        self.net_G_ema = Generator(cfg)
        accumulate(self.net_G_ema, self.net_G, 0)
        self.accum = 0.5 ** (32 / (10 * 1000))

        exp_dim = cfg.model.exp_dim
        pose_dim = cfg.model.pose_dim
        shape_dim = cfg.model.shape_dim

        # Discriminator
        from .StyleGAN2.model import ProgressiveDiscriminator

        self.net_D_stylegan = ProgressiveDiscriminator(cfg.dataset.image_size, exp_dim=exp_dim, pose_dim=pose_dim, shape_dim=shape_dim)
        self.net_D_auxgan = ProgressiveDiscriminator(cfg.model.volume_size, exp_dim=exp_dim, pose_dim=pose_dim, shape_dim=shape_dim)

        self.lr = cfg.train.lr

        self.d_reg_every = 16
        self.g_reg_every = 4

        self.flamedecoder = FlameDecoder(**cfg.model.flamedecoder)

    def forward(self, shape, c2w, uv, depth):
        output = self.net_G(shape, c2w, uv, depth, update_mean=True)
        return output

    def training_step(self, batch, bidx, optimizer_idx):
        #
        self.transform_batch(batch)

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(batch, bidx)
        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(batch, bidx)
            self.net_G.decoder.step = self.global_step  # annealing noise std

        return loss

    def generator_step(self, batch, bidx):
        return self._step(batch, bidx, for_train=True)

    def discriminator_step(self, batch, bidx):
        img_sz = self.hparams.dataset.image_size
        volume_sz = self.hparams.model.volume_size

        with torch.no_grad():
            gt = batch['image']

            c2w_fake = batch['c2w_fake']
            depth_fake = batch['depth_fake']
            uv_fake = batch['uv_fake']
            shape_fake = batch['codedict_fake']['shape']
            jaw_fake = batch['codedict_fake']['pose'][:, 3:]
            shape_fake = torch.cat([shape_fake, jaw_fake], dim=1)

            pose_real = batch['pose_real']
            exp_real = batch['codedict_real']['exp']
            jaw_real = batch['codedict_real']['pose'][:, 3:]
            exp_real = torch.cat([exp_real, jaw_real], dim=1)
            shape_real_wo_jaw = batch['codedict_real']['shape']

            gt = F.interpolate(gt, size=(img_sz, img_sz), mode='bilinear')
            gt_low = F.interpolate(gt, size=(volume_sz, volume_sz), mode='bilinear')
            gt.requires_grad_(True)
            gt_low.requires_grad_(True)

            output = self.forward(shape_fake, c2w_fake, uv_fake, depth_fake)

            pred = output['pred']
            pred.requires_grad = True

            auxgan_out = output['aux_out']
            auxgan_out.requires_grad = True

        alpha = min(1, self.global_step / 1000.0)

        # internal loss function
        def _loss(fake, real, D, r1_weight):
            pred_fake, _, _, _= D(fake, alpha)
            pred_real, pred_real_pose, pred_real_exp, pred_real_shape = D(real, alpha)
            loss_gan = d_logistic_loss(pred_real, pred_fake)

            loss_r1 = torch.zeros_like(loss_gan)
            if r1_weight > 0:
                loss_r1 = d_r1_loss(pred_real, real)
                loss_r1 = r1_weight / 2 * loss_r1 * self.d_reg_every + 0 * pred_real[0]

            loss_pose = torch.clamp(F.mse_loss(pred_real_pose, pose_real, reduction='none'), 0.0, 1.0).mean()
            loss_exp = F.mse_loss(pred_real_exp, exp_real)
            loss_shape = F.mse_loss(pred_real_shape, shape_real_wo_jaw)

            return loss_gan, loss_r1, loss_pose, loss_exp, loss_shape

        d_reg = bidx % self.d_reg_every == 0

        loss_gan_pred, loss_r1_pred, loss_pose_pred, loss_exp_pred, loss_shape_pred = _loss(pred, gt, self.net_D_stylegan, self.r1_weight_stylegan * d_reg)
        loss_gan_aux, loss_r1_aux, loss_pose_aux, loss_exp_aux, loss_shape_aux = _loss(auxgan_out, gt_low, self.net_D_auxgan, self.r1_weight * d_reg)

        loss_gan = loss_gan_pred + loss_gan_aux * self.auxgan_weight
        loss_r1 = loss_r1_pred + loss_r1_aux * self.auxgan_weight
        loss_pose = (loss_pose_pred + loss_pose_aux) * 0.5
        loss_exp = (loss_exp_pred + loss_exp_aux) * 0.5
        loss_shape = (loss_shape_pred + loss_shape_aux) * 0.5

        loss = loss_gan * self.adv_weight + \
                loss_pose * self.pose_weight + \
                loss_exp * self.exp_weight + \
                loss_shape * self.shape_weight + \
                loss_r1

        # logging
        self.log('loss_train/dis_stylegan', loss_gan_pred)
        self.log('loss_train/dis_auxgan', loss_gan_aux)
        self.log('loss_train/r1_auxgan', loss_r1_aux)
        self.log('loss_train/r1_stylegan', loss_r1_pred)
        self.log('loss_train/dis_pose_stylegan', loss_pose_pred)
        self.log('loss_train/dis_pose_auxgan', loss_pose_aux)
        self.log('loss_train/dis_exp_stylegan', loss_exp_pred)
        self.log('loss_train/dis_exp_auxgan', loss_exp_aux)
        self.log('loss_train/dis_shape_stylegan', loss_shape_pred)
        self.log('loss_train/dis_shape_aux', loss_shape_aux)

        # #logging discriminator accuracy
        # real_acc = pred_real[0][-1] > 0.5
        # real_acc = real_acc.float().mean()
        # fake_acc = pred_fake[0][-1] <= 0.5
        # fake_acc = fake_acc.float().mean()
        # self.log('acc/real', real_acc)
        # self.log('acc/fake', fake_acc)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, bidx):
        self.transform_batch(batch)

        _ = self._step(batch, bidx, for_train=False)

    def _step(self, batch, bidx, for_train=True):
        # img_sz = self.stage['image_size']
        gt = batch['image']

        c2w = batch['c2w_fake']
        pose = batch['pose_fake']
        uv = batch['uv_fake']
        depth = batch['depth_fake']
        jaw = batch['codedict_fake']['pose'][:, 3:]
        shape_wo_jaw = batch['codedict_fake']['shape']
        shape = torch.cat([shape_wo_jaw, jaw], dim=1)
        exp = batch['codedict_fake']['exp']
        exp = torch.cat([exp, jaw], dim=1)

        output = self(shape, c2w, uv, depth)

        pred = output['pred']
        pred_depth = output['depth']
        auxgan_out = output['aux_out']
        face_alpha = output['face_alpha']

        alpha = min(1, self.global_step / 1000.0)

        def _loss(fake, D, topk):
            pred_fake, pred_fake_pose, pred_fake_exp, pred_fake_shape = D(fake, alpha)

            if topk < 1.0:
                topk_num = math.ceil(topk * pred_fake.shape[0])
                pred_fake = torch.topk(pred_fake, topk_num, dim=0).values
                loss_gan = F.softplus(-pred_fake).mean()
            else:
                loss_gan = g_nonsaturating_loss(pred_fake)

            loss_pose = torch.clamp(F.mse_loss(pred_fake_pose, pose, reduction='none'), 0.0, 1.0).mean()
            loss_exp = F.mse_loss(pred_fake_exp, exp)
            loss_shape = F.mse_loss(pred_fake_shape, shape_wo_jaw)

            return loss_gan, loss_pose, loss_exp, loss_shape

        loss_gan_pred, loss_pose_pred, loss_exp_pred, loss_shape_pred = _loss(pred, self.net_D_stylegan, 1.0)

        topk_percentage = max(0.99 ** (self.global_step/2000), 0.6)
        loss_gan_aux, loss_pose_aux, loss_exp_aux, loss_shape_aux = _loss(auxgan_out, self.net_D_auxgan, topk_percentage)

        loss_gan = loss_gan_pred + loss_gan_aux * self.auxgan_weight
        loss_pose = (loss_pose_pred + loss_pose_aux) * 0.5
        loss_exp = (loss_exp_pred + loss_exp_aux) * 0.5
        loss_shape = (loss_shape_pred + loss_shape_aux) * 0.5 

        resize_factor = int(self.hparams.dataset.image_size / self.hparams.model.volume_size)
        pred_low = F.avg_pool2d(pred, kernel_size=resize_factor, stride=resize_factor)
        loss_aux = F.mse_loss(auxgan_out, pred_low)

        # face alpha loss
        face_mask = (uv != 0).all(1, keepdim=True).float()
        face_mask = F.adaptive_avg_pool2d(face_mask, output_size=face_alpha.shape[-1])
        loss_alpha = -torch.log(torch.clamp(face_alpha, min=1e-08, max=1))
        loss_alpha = torch.mean(loss_alpha * face_mask)

        loss =  loss_gan * self.adv_weight +\
                loss_pose * self.pose_weight + \
                loss_exp * self.exp_weight +\
                loss_shape * self.shape_weight +\
                loss_aux * self.aux_weight +\
                loss_alpha * self.alpha_weight

        # logging
        log_name = 'train' if for_train else 'val'
        self.log(f'loss_{log_name}/full', loss.item(), prog_bar=True, logger=True)
        self.log(f'loss_{log_name}/adv', loss_gan)
        self.log(f'loss_{log_name}/aux', loss_aux)
        self.log(f'loss_{log_name}/pose_stylegan', loss_pose_pred)
        self.log(f'loss_{log_name}/pose_auxgan', loss_pose_aux)
        self.log(f'loss_{log_name}/exp_stylegan', loss_exp_pred)
        self.log(f'loss_{log_name}/exp_auxgan', loss_exp_aux)
        self.log(f'loss_{log_name}/shape_stylegan', loss_shape_pred)
        self.log(f'loss_{log_name}/shape_auxgan', loss_shape_aux)
        self.log(f'loss_{log_name}/alpha', loss_alpha)

        if bidx % 50 == 0:
            vis_pred = pred * 0.5 + 0.5
            vis_gt = gt * 0.5 + 0.5
            vis_pred = make_grid(vis_pred)
            vis_gt = make_grid(vis_gt)

            vis_depth, d_min, d_max = self.make_depth_vis(pred_depth)
            self.log('depth/min', d_min)
            self.log('depth/max', d_max)

            self.logger.experiment.add_image(f'{log_name}/pred', vis_pred, self.global_step)
            self.logger.experiment.add_image(f'{log_name}/gt', vis_gt, self.global_step)
            self.logger.experiment.add_image(f'{log_name}/depth', vis_depth, self.global_step)

            # if self.use_aux:
            vis_aux = auxgan_out * 0.5 + 0.5
            vis_aux = make_grid(vis_aux)
            self.logger.experiment.add_image(f'{log_name}/aux', vis_aux, self.global_step)

            # ema
            self.net_G_ema.eval()
            with torch.no_grad():
                output_ema = self.net_G_ema(shape, c2w, uv, depth, truncation=0.5, update_mean=False)
            pred_ema = output_ema['pred']
            vis_pred_ema = pred_ema * 0.5 + 0.5
            vis_pred_ema = make_grid(vis_pred_ema)

            vis_depth, _, _ = self.make_depth_vis(output_ema['depth'])

            self.logger.experiment.add_image(f'{log_name}/pred_ema', vis_pred_ema, self.global_step)
            self.logger.experiment.add_image(f'{log_name}/depth_ema', vis_depth, self.global_step)

        return loss

    def train_dataloader(self):
        dataloader_train = DataLoader(
            self.dataset_train,
            num_workers=self.hparams.dataloader.num_workers,
            batch_size=self.hparams.dataloader.batch_size,
            worker_init_fn=_set_worker_numpy_seed, shuffle=True, drop_last=True)

        return dataloader_train

    def val_dataloader(self):
        dataloader_val = DataLoader(
            self.dataset_val,
            num_workers=self.hparams.dataloader.num_workers,
            batch_size=self.hparams.dataloader.batch_size,
            worker_init_fn=_set_worker_numpy_seed, shuffle=False, drop_last=True)

        return dataloader_val

    def make_depth_vis(self, depth):
        depth = torch.where(torch.isnan(depth), torch.zeros_like(depth), depth)
        d_min = torch.min(depth)
        d_max = torch.max(depth)
        depth = (depth - d_min) / max(d_max-d_min, 1e-8)  # normalize to 0~1
        vis_depth = make_grid(depth)
        return vis_depth, d_min, d_max

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx=optimizer_idx, optimizer_closure=optimizer_closure, on_tpu=on_tpu, using_native_amp=using_native_amp, using_lbfgs=using_lbfgs)

        if optimizer_idx == 0:
            with torch.no_grad():
                accumulate(self.net_G_ema, self.net_G, self.accum)

    def configure_optimizers(self):
        #
        g_reg_ratio = 1
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        G_param_decoder, G_param_stylegan = self.net_G.get_generator_params_for_optim()
        kwargs_net_G_face = {'lr': self.lr['g_decoder'], 'betas': (0.0, 0.9)}

        G_optim = optim.Adam([
            {'params': G_param_decoder, **kwargs_net_G_face},  # generator decoder
            {'params': G_param_stylegan, 'lr': self.lr['g_stylegan'] * g_reg_ratio, 'betas': (0.0 ** g_reg_ratio, 0.99 ** g_reg_ratio)}  # stylegan
        ])

        D_optim = [
            {'params': self.net_D_stylegan.parameters(), 'lr': self.lr['d_stylegan'] * d_reg_ratio, 'betas': (0.9 ** d_reg_ratio, 0.99 ** d_reg_ratio)},
        ]
        kwargs_net_D_auxgan = {
            'lr': self.lr['d_auxgan'] * d_reg_ratio,
            'betas': (0.0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        }
        D_optim.append({'params': self.net_D_auxgan.parameters(), **kwargs_net_D_auxgan})

        D_optim = optim.Adam(D_optim)

        return [G_optim, D_optim], []

    def transform_batch(self, batch):
        with torch.no_grad():
            out = self.flamedecoder(batch['codedict_fake'], batch['bbox_fake'])

        batch['uv_fake'] = out['uv']
        batch['depth_fake'] = out['depth']
        batch['c2w_fake'] = out['c2w']

        # get pose
        # pose of the input image
        quat = matrix_to_quaternion(batch['c2w_real'][:, :3, :3])
        t = batch['c2w_real'][:, :2, 3]
        pose_real = torch.cat([quat, t], dim=-1)
        batch['pose_real'] = pose_real

        # pose of the augmented uv, depth map
        quat = matrix_to_quaternion(out['c2w'][:, :3, :3])
        t = out['c2w'][:, :2, 3]
        pose_fake = torch.cat([quat, t], dim=-1)
        batch['pose_fake'] = pose_fake

        return batch

# == pipeline class ended == #


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
    model1.mean_w.mul_(decay).add_(model2.mean_w, alpha=1 - decay)


def _set_worker_numpy_seed(worker_idx):
    #
    seed = get_worker_info().seed % 2**32
    np.random.seed(seed)
    random.seed(seed)
