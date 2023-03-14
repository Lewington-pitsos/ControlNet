import numpy as np
import random
import torch

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger, ZeroConvLogger
from cldm.model import create_model, load_state_dict
import argparse
from wds_load import load_laion


parser = argparse.ArgumentParser()

parser.add_argument('--model_config_path', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--test_url', type=str)
parser.add_argument('--resume_path', type=str, default='./models/control_sd21_ini.ckpt')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_logger_freq', type=int, default=300)
parser.add_argument('--zc_logger_freq', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--sd_locked', type=bool, default=True)
parser.add_argument('--only_mid_control', type=bool, default=False)
parser.add_argument('--accumulate_grad_batches', type=int, default=None)
parser.add_argument('--max-steps', type=int, default=4000)
parser.add_argument('--experiment_name', type=str, default='controlnet-ablation-test')

args = parser.parse_args()

def create_hparam_model(config_path, **kwargs):
    model=create_model(config_path)
    for key, value in kwargs.items():
        model.hparams[key] = value

    model.learning_rate = model.hparams['learning_rate']

    return model

def perform_training_run(args):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_hparam_model(
        args.model_config_path,
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        train_url=args.train_url, 
        test_url=args.test_url,
        hint_proportion=0.35,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps
    ).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    print('this is the method', model.on_train_start)

    train_dl, test_dl = load_laion(
        model.hparams['batch_size'], 
        model.hparams['train_url'], 
        model.hparams['test_url'],
        model.hparams['hint_proportion'], 
    )

    img_logger = ImageLogger(batch_frequency=args.img_logger_freq)
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
    zc_logger = ZeroConvLogger(args.zc_logger_freq)
    wandb_logger = WandbLogger(project=args.experiment_name)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        precision=32, 
        callbacks=[img_logger, zc_logger, checkpoint_callback], 
        logger=[wandb_logger], 
        accumulate_grad_batches=model.hparams['accumulate_grad_batches'],
        log_every_n_steps=args.zc_logger_freq,
        max_steps=model.hparams['max_steps'],
        val_check_interval=500
    )

    wandb.init(project=args.experiment_name)

    trainer.fit(model, train_dl, test_dl)

if __name__ == '__main__':
    perform_training_run(args)