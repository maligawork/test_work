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


@hydra.main(version_base=None, config_path='../configs', config_name='test')
def run_test(cfg: DictConfig):
    """Run the test pipeline with params from the config-file.
        Args
            cfg (DictConfig): read more in configs-folder.
    """
    if os.getenv('LOCAL_RANK', '0') == '0':
        print(OmegaConf.to_yaml(cfg))
        output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
        os.environ['hydra_dir'] = output_dir
    else:
        output_dir = os.environ['hydra_dir']

    folders = output_dir.split('/')
    for idx, folder in enumerate(folders):
        if folder == 'experiments_results':
            output_dir = os.path.join('..', '/'.join(folders[idx:]))
            break

    # define dataset
    n_cpu = os.cpu_count()
    dataset = TriangleSquareDataModule(cfg.data, batch_size=cfg.hyp.batch_size, num_workers=n_cpu)
    dataset.setup('test')
   
    # define model
    model_creator = SiameseModelsCreator(**cfg.model)
    
    # define lightning model
    od_model = SiameseLightningModel(
        model=model_creator.model,
        loss_fn=model_creator.loss_fn,
        optimizer_params=cfg.hyp.optim,
        lr_scheduler_params=cfg.hyp.lr_scheduler,
        metrics_threshold=cfg.hyp.threshold,
    )

    # define checkpoints
    checkpoint_dirpath = os.path.join(output_dir, 'checkpoints')
    tb_logger = TensorBoardLogger(save_dir=output_dir, name='tensorboard_logs', version=0)

    ckpt_path = None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dirpath, '*.ckpt')))
    if len(checkpoints) != 0:
        ckpt_path = checkpoints[-1]
    else:
        raise ValueError('No checkpoints found!')
    print(f'Loading checkpoint: {ckpt_path}')

    trainer = pl.Trainer(max_epochs=cfg.hyp.max_epochs,
                         logger=tb_logger,
                         devices=cfg.hyp.n_devices,
    )

    print('TEST START!')
    test_metrics = trainer.test(model=od_model, ckpt_path=ckpt_path, datamodule=dataset, verbose=False)
    print(f'Test metrics: {test_metrics}')
    print('TEST FINISH!')


if __name__ == '__main__':
    run_test()