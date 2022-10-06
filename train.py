import argparse
import random

import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from torch.utils.data import get_worker_info

from dataset.ffhq import FFHQDataset
from models.pipeline_dev import Pipeline

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--cfg', type=str)
args = parser.parse_args()


def _set_worker_numpy_seed(worker_idx):
    #
    seed = get_worker_info().seed % 2**32
    np.random.seed(seed)
    random.seed(seed)


def main():
    cfg = OmegaConf.load(args.cfg)

    model = Pipeline(cfg)

    dataset_train = FFHQDataset(**cfg.dataset, split='train')
    dataset_val = FFHQDataset(**cfg.dataset, split='val')

    model.dataset_train = dataset_train
    model.dataset_val = dataset_val

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor='loss_val/full',
        mode='min'
    )

    progress_callback = ProgressBar(cfg.train.progress_refresh_step)

    save_dir = cfg.save_dir
    if save_dir is None:
        save_dir = './'
    train_logger = pl_loggers.TensorBoardLogger(
        save_dir=save_dir,
        name='logs',
        version=None)

    trainer = pl.Trainer(
        logger=train_logger,
        callbacks=[checkpoint_callback, progress_callback],
        accelerator=cfg.train.backend,
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        log_every_n_steps=30,
        num_sanity_val_steps=1,
        gradient_clip_val=10,
        resume_from_checkpoint=cfg.checkpoint_path,
        reload_dataloaders_every_n_epochs=1
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
