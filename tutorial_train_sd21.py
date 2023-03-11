from share import *
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger, ZeroConvLogger
from cldm.model import create_model, load_state_dict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--resume_path', type=str, default='./models/control_sd21_ini.ckpt')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--logger_freq', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--sd_locked', type=bool, default=True)
parser.add_argument('--only_mid_control', type=bool, default=False)
parser.add_argument('--accumulate_grad_batches', type=int, default=None)
parser.add_argument('--experiment_name', type=str, default='controlnet-ablation-test')

args = parser.parse_args()

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
img_logger = ImageLogger(batch_frequency=args.logger_freq)
zc_logger = ZeroConvLogger()
wandb_logger = pl.loggers.WandbLogger(project=args.experiment_name)
trainer = pl.Trainer(
    gpus=1, 
    precision=32, 
    callbacks=[img_logger, zc_logger], 
    logger=[wandb_logger], 
    accumulate_grad_batches=args.accumulate_grad_batches
)

wandb.init(project=args.experiment_name)

# Train!
trainer.fit(model, dataloader)
