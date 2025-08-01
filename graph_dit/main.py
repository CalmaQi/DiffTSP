# These imports are tricky because they use c++, do not move them
import os, shutil
import warnings

import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import utils
from datasets import kgdata,kgdata_lightning
from diffusion.difftest import Diffusion
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = TensorBoardLogger(save_dir="./logs")

torch.set_float32_matmul_precision("medium")
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

from pytorch_lightning.callbacks import Callback

def remove_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    batch_size = cfg.train.batch_size
    model = Graph_DiT.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.train.batch_size = batch_size
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model

def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == "discrete":
        model = Graph_DiT.load_from_checkpoint(
            resume_path, **model_kwargs
        )
    else:
        raise NotImplementedError("Unknown model")

    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


@hydra.main(
     config_path="../configs", config_name="config"
)
def main(cfg: DictConfig):

    
    data_module = kgdata_lightning.TransductiveDataModule(cfg)
    test_triples = data_module.data_object.test_triplets
    seen_triples = data_module.data_object.seen_triplets
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="valid/f1",
        mode="max",
        filename="{epoch}-{valid/f1:.3f}",
        save_top_k=1,
    )
  
    model = Diffusion(config=cfg,test_triples=test_triples)
 
    model.test_triples = test_triples
    model.seentriples = seen_triples
    model.T = cfg.model.diffusion_steps
    
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        accelerator="gpu"
        if torch.cuda.is_available() and cfg.general.gpus > 0
        else "cpu",
        devices=cfg.general.gpus
        if torch.cuda.is_available() and cfg.general.gpus > 0
        else None,
        max_epochs=cfg.train.n_epochs,
        enable_checkpointing=True,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        strategy="ddp" if cfg.general.gpus > 1 else "auto",
        enable_progress_bar=cfg.general.enable_progress_bar,
        callbacks=[model_checkpoint],
        reload_dataloaders_every_n_epochs=0,
        logger=logger,
        num_sanity_val_steps=cfg.general.num_sanity_val_steps,
    )

    if not cfg.general.test_only:

        trainer.fit(model, datamodule=data_module)
        trainer.validate(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=cfg.general.test_only)

if __name__ == "__main__":
    main()
