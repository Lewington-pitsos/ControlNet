import wandb
import torch
from pytorch_lightning.callbacks import Callback

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            images = wandb.Image(images)
            wandb.log({"examples": images})

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")


class ZeroConvLogger(Callback):
  def on_train_batch_end(self, trainer, pl_module):
      with torch.no_grad():
        count = 0
        bias_mean = 0
        bias_std = 0

        weight_mean = 0
        weight_std = 0

        for i, c in enumerate(pl_module.control_model.zero_convs):
        # print(type(c))
          layer = c[0]
          # print(type(layer))

          bias_mean += layer.bias.mean()
          bias_std += layer.bias.std()

          weight_mean += layer.weight.mean()
          weight_std += layer.weight.std()

          wandb.log({
              f'{i}-zc-bias-mean': layer.bias.mean(),
              f'{i}-zc-bias-std': layer.bias.std(),
              f'{i}-zc-bias-min': layer.bias.min(),
              f'{i}-zc-bias-max': layer.bias.max(),

              f'{i}-zc-weight-mean': layer.weight.mean(),
              f'{i}-zc-weight-std': layer.weight.std(),
              f'{i}-zc-weight-min': layer.weight.min(),
              f'{i}-zc-weight-max': layer.weight.max(),
          })
          count += 1

        wandb.log({
            'zc-all-biases-mean': bias_mean / count,
            'zc-all-biases-std': bias_std / count,
            'zc-all-weights-mean': weight_mean / count,
            'zc-all-weights-std': weight_std / count,
        })
