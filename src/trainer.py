from copy import copy
import wandb
import os
import re 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from loguru import logger

from .modeling import Model
from .utils import get_group_name, init_wandb

def train(params, data_module, root_dir = None):

    # Create a PyTorch Lightning trainer with the generation callback
    if root_dir is None : root_dir = os.path.join(params.log_dir, params.exp_id) 
    pp = vars(params)
    trainer_config = {
        "max_epochs": params.max_epochs,
        "default_root_dir" : root_dir,

        # "limit_train_batches" : pp.get("limit_train_batches", 1.0), 
        # "limit_val_batches" : pp.get("limit_val_batches", 1.0),
        # "limit_test_batches": pp.get("limit_test_batches", 1.0),

        "accelerator" : params.accelerator,
        "devices" : params.devices,
        #"reload_dataloaders_every_n_epochs" : True,
        "weights_summary":"full", # "top", None,

        # "log_every_n_steps" : max(len(train_loader) // params.batch_size, 0),
        # "weights_save_path" : os.path.join(root_dir, "weights"),
        # "auto_scale_batch_size" : True, # None
        # "auto_select_gpus" : True,
        # "auto_lr_find": True,
        # "benchmark" : False,
        # "deterministic" : True,
        # "val_check_interval" : 1.,
        # "accumulate_grad_batches" : False,
        # "strategy": "ddp", # "ddp_spaw"
    }

    validation_metrics = params.validation_metrics
    mode = (lambda s : "min" if 'loss' in s else 'max')(validation_metrics)
    early_stopping_callback = EarlyStopping(
        monitor=validation_metrics, patience=params.early_stopping_patience, verbose=False, strict=True,
        mode = mode
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath=root_dir,
            save_weights_only=True,
            filename="{epoch}-{%s:.4f}"%validation_metrics,
            mode = mode,
            monitor=validation_metrics,
            save_top_k=params.save_top_k,
            save_last=True,
            every_n_epochs=getattr(params, 'every_n_epochs', 1)
    )

    trainer_config["callbacks"] = [
        early_stopping_callback, 
        model_checkpoint_callback,
        LearningRateMonitor("epoch")
    ]

    trainer = pl.Trainer(**trainer_config)
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = root_dir + params.model_name
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model %s, loading..."%pretrained_filename)
        model = Model.load_from_checkpoint(pretrained_filename)
        print(model)
    else:
        logger.info("Training start")
        # Initialize wandb
        if params.group_name is None : params.group_name = get_group_name(params, group_vars = params.group_vars)
        init_wandb(params.use_wandb, wandb_project = params.wandb_project, group_name = params.group_name, wandb_entity = params.wandb_entity)

        model = Model(params)
        print(model)
        trainer.fit(model, datamodule = data_module, ckpt_path=params.checkpoint_path)
        model = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        #try : wandb.finish(exit_code = None, quiet = None)
        #except : pass
        logger.info("Training end")

    # Test best model on validation set
    logger.info("Testing start")
    val_result = trainer.validate(model, datamodule = data_module, verbose=False)
    data_module.val_dataloader = data_module.train_dataloader
    train_result = trainer.validate(model, datamodule = data_module, verbose=False)
    logger.info("Testing end")

    result = {"train": train_result, "val": val_result}
    for k1, v1 in copy(result).items() :
        #for k2 in v1[0] : result[k1][0][k2.replace("val", k1)] = round(result[k1][0].pop(k2), 4)
        result[k1] = {k2.replace("val", k1): round(result[k1][0][k2], 4) for k2 in v1[0]}
    
    return model, result