def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()

from share import *
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger, ZeroConvLogger
from cldm.model import create_model, load_state_dict
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_config_path', type=str)
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

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
pl_module = create_model(args.model_config_path).cpu()
pl_module.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
pl_module.learning_rate = args.learning_rate
pl_module.sd_locked = args.sd_locked
pl_module.only_mid_control = args.only_mid_control


with torch.no_grad():
    count = 0
    weight_mean = 0
    weight_std = 0
    weight_frobenius_norm = 0
    
    for i, c in enumerate(pl_module.control_model.zero_convs):
        layer = c[0]

        weight_mean += layer.weight.mean()
        weight_std += layer.weight.std()
        weight_frobenius_norm += torch.norm(layer.weight)

        print({
            f'zc-{i}': {                
                'zc-weight-std': layer.weight.std(),
                'zc-weight-mean': layer.weight.mean(),
                'zc-weight-frob': torch.norm(layer.weight),
            }
        })
        count += 1

        print({
            'zc-all-weights-mean': weight_mean / count,
            'zc-all-weights-frobenius': weight_frobenius_norm / count,
            'zc-all-weights-std': weight_std / count,
        })
