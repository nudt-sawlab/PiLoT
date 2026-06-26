
# from pytorch_lightning.callbacks import ProgressBar
# from pytorch_lightning.callbacks import ProgressBarBase 
import pytorch_lightning as pl

import os
import torch
import warnings
from ..dataset import get_dataset
from ..utils.lightening_utils import MyLightningLogger,convert_old_model, load_model_weight
from ..trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin

def train(cfg):

    logger = MyLightningLogger('PiLoT', cfg.save_dir)
    logger.dump_cfg(cfg, 'train_cfg.yml')

    logger.info("Setting up data...")
    dataset = get_dataset(cfg.data.name)(cfg.data)
    if 'data1' not in cfg:
        train_data_loader = dataset.get_data_loader('train')
        val_data_loader = dataset.get_data_loader('val')
        # train_data_loader = val_data_loader
    else:
        from torch.utils.data import DataLoader
        from ..dataset.base_dataset import collate, worker_init_fn
        data_loaders = []
        for split in ['train', 'val']:
            dataset = get_dataset(cfg.data.name)(cfg.data).get_dataset(split)
            dataset_1 = get_dataset(cfg.data1.name)(cfg.data1).get_dataset(split)
            new_dataset = torch.utils.data.ConcatDataset([dataset, dataset_1])
            batch_size = cfg.data[split+'_batch_size_per_gpu']
            num_workers = cfg.data.get('workers_per_gpu', batch_size)
            shuffle = split == 'train'
            data_loaders.append(DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle,
                                           sampler=None, pin_memory=True, collate_fn=collate,
                                           num_workers=num_workers, worker_init_fn=worker_init_fn))
        train_data_loader = data_loaders[0]
        val_data_loader = data_loaders[1]

    logger.info("Creating model...")
    task = Trainer(cfg, train_data_loader)
    
    # TODO: Load model
    if "load_model" in cfg:
        ckpt = torch.load(cfg.load_model, map_location='cpu')
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.load_model))

    # model_resume_path = (
    #     os.path.join(cfg.save_dir, "model_last.ckpt")
    #     if "resume" in cfg
    #     else None
    # )
    model_resume_path = (
        os.path.join(cfg.resume, "model_last.ckpt")
        if "resume" in cfg
        else None
    )
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.trainer.total_epochs,
        gpus=cfg.device.gpu_ids,
        devices=len(cfg.device.gpu_ids),
        check_val_every_n_epoch=cfg.trainer.val_intervals,
        accelerator="gpu",  # "ddp",
        strategy="ddp", # "ddp_find_unused_parameters_false", # "ddp",
        log_every_n_steps=cfg.trainer.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        # plugins=DDPPlugin(find_unused_parameters=False),
        logger=logger,
        benchmark=True,
        # deterministic=True,
        # accumulate_grad_batches=4,
        # callbacks=[ProgressBarBase()],  # disable tqdm bar
        callbacks=[],
        enable_progress_bar=False

    )

    trainer.fit(model=task, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)