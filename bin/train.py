import os
import sys
import glob

sys.path.append('../')

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import pytorch_lightning as pl

from pipeline.data import TriangleSquareDataModule
from pipeline.modules import SiameseModelsCreator
from pipeline.trainers import SiameseLightningModel
from pipeline.callbacks import (
    ModelCheckpoint, 
    TensorBoardLogger, 
    LearningRateMonitor, 
    calculate_accumulate_steps,
)

torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path='../configs', config_name='train')
def run_train(cfg: DictConfig):
    """Run the training/validation pipeline with params from the config-file.
        Args
            cfg (DictConfig): read more in configs-folder.
    """
    if os.getenv('LOCAL_RANK', '0') == '0':
        print(OmegaConf.to_yaml(cfg))
        output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        os.environ['hydra_dir'] = output_dir
    else:
        output_dir = os.environ['hydra_dir']

    # define dataset
    n_cpu = os.cpu_count()
    dataset = TriangleSquareDataModule(cfg.data, batch_size=cfg.hyp.batch_size, num_workers=n_cpu)
    dataset.setup('fit')
    if cfg.hyp.lr_scheduler['scheduler_type'] == 'cosine_annealing':
        cfg.hyp.lr_scheduler['t_max'] = cfg.hyp.max_epochs * len(dataset.train_dataloader())
   
    # define model
    model_creator = SiameseModelsCreator(**cfg.model)


    calculate_accumulate_steps
    accumulate_grad_batches, weight_decay = calculate_accumulate_steps(
        batch_size=cfg.hyp.batch_size,
        weight_decay=cfg.hyp.optim.weight_decay,
        nominal_batch_size=cfg.hyp.nominal_batch_size,
    )
    cfg.hyp.optim.weight_decay = weight_decay
    
    # define checkpoints
    checkpoint_dirpath = os.path.join(output_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=cfg.hyp.save_top_k,
        monitor='valid_loss',
        mode='min',
        save_last=True,
        dirpath=checkpoint_dirpath,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    tb_logger = TensorBoardLogger(save_dir=output_dir, name='tensorboard_logs', version=0)

    ckpt_path = None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dirpath, '*.ckpt')))
    if len(checkpoints) != 0:
        ckpt_path = checkpoints[-1]

    # define lightning model
    od_model = SiameseLightningModel(
        model=model_creator.model,
        loss_fn=model_creator.loss_fn,
        optimizer_params=cfg.hyp.optim,
        lr_scheduler_params=cfg.hyp.lr_scheduler,
        metrics_threshold=cfg.hyp.threshold,
    )

    # define trainer
    trainer = pl.Trainer(max_epochs=cfg.hyp.max_epochs,
                         accumulate_grad_batches=accumulate_grad_batches,
                         callbacks=[checkpoint_callback, lr_monitor],
                         logger=tb_logger,
                         strategy=cfg.hyp.strategy,
                         precision=cfg.hyp.precision,
                         log_every_n_steps=cfg.hyp.log_every_n_steps,
                         devices=cfg.hyp.n_devices,
    )

    if not ckpt_path:
        print('TRAINING START!')
        trainer.fit(
            model=od_model,
            datamodule=dataset,
        )
    else:
        print('TRAINING RESUME!')
        trainer.fit(
            model=od_model,
            datamodule=dataset,
            ckpt_path=ckpt_path,
        )

    print('TRAINING FINISH!')
    
    print(f'Best checkpoint: {checkpoint_callback.best_model_path}')

    print('TEST START!')
    trainer.devices = 1
    valid_metrics = trainer.validate(ckpt_path='best', datamodule=dataset, verbose=False)
    print(f'Validation metrics: {valid_metrics}')
    test_metrics = trainer.test(ckpt_path='best', datamodule=dataset, verbose=False)
    print(f'Test metrics: {test_metrics}')
    print('TEST FINISH!')


if __name__ == '__main__':
    run_train()