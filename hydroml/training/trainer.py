from typing import List, Optional

import torch
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from hydroml.config.config import Config


def get_callbacks(config: Config) -> List:
    """Get training callbacks for PyTorch Lightning.
    
    Args:
        config (Config): Configuration object containing training parameters
        
    Returns:
        List: List of callbacks including learning rate monitoring and model checkpointing
    """
    # Monitor learning rate changes during training
    lr_logger = LearningRateMonitor()
    
    # Save model checkpoints based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.current_path / config.version,
        monitor='val_loss',
        filename='checkpoint_{epoch:02d}',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_top_k=config.save_top_k,
        save_last=True,
        mode='min',
        auto_insert_metric_name=True
    )
    
    return [lr_logger, checkpoint_callback]


def get_trainer(config: Config, strategy: Optional[str] = None) -> Trainer:
    """Create PyTorch Lightning trainer with specified configuration.
    
    Args:
        config (Config): Configuration object containing training parameters
        strategy (Optional[str]): Training strategy. Defaults to DDP with unused parameter finding
        
    Returns:
        Trainer: Configured PyTorch Lightning trainer
    """
    # Set default distributed training strategy if none provided
    accelerator = config.device
    if strategy is None:
        strategy = 'ddp_find_unused_parameters_true'
    
    # Configure tensorboard logging
    logger = TensorBoardLogger(config.current_path, name='', version=config.version)

    print(logger.log_dir)
    
    # Set up trainer kwargs
    trainer_config = {
        'max_epochs': config.max_epochs,
        'logger': logger,
        'accelerator': accelerator,
        'enable_model_summary': True,
        'gradient_clip_val': config.gradient_clip_val,
        'callbacks': get_callbacks(config),
        'check_val_every_n_epoch': config.check_val_every_n_epoch,
        'enable_progress_bar': config.enable_progress_bar,
    }
    
    # Configure CUDA-specific settings if available
    if torch.cuda.is_available() and accelerator != 'cpu' :
        trainer_config['strategy'] = strategy
        torch.set_float32_matmul_precision('medium')
        #trainer_config['strategy'] = 'ddp'
        #trainer_config['strategy'] = 'auto'
    
    return Trainer(**trainer_config)