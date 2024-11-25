import copy
from utils.utils import set_requires_grad,update_moving_average
from torch.utils.data import DataLoader
from network.model_utils import EMA,make_sym,noise_sym,noise_sym_like
from network.data_loader import ImageDataset
from network.model import myDiffusion

from pathlib import Path
from torch.optim import AdamW,Adam
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import random
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

class DiffusionModel(LightningModule):
    def __init__(
        self,
        img_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = myDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose)

        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.img_folder = img_folder
        self.data_class = data_class
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out

        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()
        # print("os.cpu_count(): ",os.cpu_count(), "  ", self.num_workers)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        _dataset = ImageDataset(resolution=self.image_size,
                                data_folder=self.img_folder,)
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        print("data path: ", self.img_folder, "dataset num: ", _dataset.__len__(), "batch num: ", self.iterations)
        return dataloader

    def training_step(self, batch, batch_idx):
        image_features = None
        projection_matrix = None
        kernel_size = None
        text_feature = None
        
        img=batch["img"]
        cond=batch["cond"]
        bdr=batch["bdr"]
        print(img.shape, cond.shape, bdr.shape)
        loss = self.model.training_loss(
            img, image_features, text_feature, projection_matrix, kernel_size=kernel_size, cond=cond,bdr=bdr).mean()
        print("loss: ",loss)
        self.log("loss", loss.clone().detach().item(), prog_bar=True)

        opt = self.optimizers()
        # opt = self.configure_optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.update_EMA()

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch)
        return super().on_train_epoch_end()


if __name__ == '__main__':
    model = DiffusionModel(
        results_folder='./results/model', 
        img_folder = "D:/Release Data/bdr_data_sim/data1/",
        data_class = "chair",
        batch_size = 4,
        lr = 2e-4,
        image_size = 64,
        noise_schedule = "linear",
        base_channels = 32,
        optimizier = "adam",
        attention_resolutions = "4, 8",
        with_attention = True,
        num_heads = 4,
        dropout = 0.1,
        ema_rate = 0.999,
        verbose = False,
        save_every_epoch = 20,
        kernel_size = 2.0,
        training_epoch = 20,
        gradient_clip_val = 1.,
        debug = False,
        image_feature_drop_out = 0.1,
        view_information_ratio = 2.0,
        data_augmentation = False,
        vit_global = False,
        vit_local = True,
        split_dataset = False,
        elevation_zero = False,
        detail_view = False
        )

    from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch

    try:
        log_dir = get_tensorboard_dir()
    except Exception as e:
        log_dir = './results/model'

    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    from pytorch_lightning import loggers as pl_loggers

    find_unused_parameters = True
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    from pytorch_lightning.callbacks import ModelCheckpoint
    save_last: bool = True
    save_every_epoch: int = 20
    checkpoint_callback = ModelCheckpoint(
        monitor="current_epoch",
        dirpath='./results/model',
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=save_last,
        every_n_epochs=save_every_epoch,
        mode="max",
    )

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      strategy=DDPPlugin(
                          find_unused_parameters=find_unused_parameters),
                      logger=tb_logger,
                      max_epochs=20,
                      log_every_n_steps=10,
                      callbacks=[checkpoint_callback])

    trainer.fit(model)
    # dataset1 = model.train_dataloader()
    # batch = next(iter(dataset1))
    # # dataset1[0]
    # model.training_step(batch, 2)