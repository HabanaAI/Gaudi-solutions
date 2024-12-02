#!/usr/bin/env python
# coding: utf-8

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Brain tumor 3D segmentation with MONAI on Gaudi and Pytorch Lightning
# Adapted from: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
# to demonstrate how to run Monai models on Gaudi with Pytorch Lightning

# The dataset comes from http://medicaldecathlon.com/.
# Target: Gliomas segmentation necrotic/active tumour and oedema
# Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)
# Size: 750 4D volumes (484 Training + 266 Testing)
# Source: BRATS 2016 and 2017 datasets.
# Challenge: Complex and heterogeneously-located targets

import argparse
import time

import habana_frameworks.torch.core as htcore
import monai
import torch
import torch.distributed as dist
from lightning import Trainer
from lightning.pytorch import LightningDataModule
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_habana.pytorch.accelerator import HPUAccelerator
from lightning_habana.pytorch.plugins.precision import HPUPrecisionPlugin
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism


def gaudi_get_dist_device():
    """
    Get the expected target device in the native PyTorch distributed data parallel.
    For NCCL backend, return GPU device of current process.
    For HCCL backend, return HPU device of current process.
    For GLOO backend, return CPU.
    For any other backends, return None as the default, tensor.to(None) will not change the device.

    """
    if dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if backend == "hccl" and torch.hpu.is_available():
            return torch.device(f"hpu:{torch.hpu.current_device()}")
        if backend == "gloo":
            return torch.device("cpu")
    return None


def adapt_monai_to_gaudi():
    """
    Replaces some Monai' methods for equivalent methods optimized
    for Gaudi.
    """
    monai.utils.dist.get_dist_device = gaudi_get_dist_device


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


class BratsDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=1, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage='validate'):
        print(f"setup data: {stage}", flush=True)
        if stage == 'fit' or stage == 'validate':
            train_transform = Compose(
                [
                    # load 4 Nifti images and stack them together
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                    RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ]
            )
            self.train_ds = DecathlonDataset(
                root_dir=self.root_dir,
                task="Task01_BrainTumour",
                transform=train_transform,
                section="training",
                download=False,
                cache_rate=0.0,
                num_workers=self.num_workers,
            )
            val_transform = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys=["image", "label"]),
                    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    Spacingd(
                        keys=["image", "label"],
                        pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest"),
                    ),
                    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ]
            )
            self.val_ds = DecathlonDataset(
                root_dir=self.root_dir,
                task="Task01_BrainTumour",
                transform=val_transform,
                section="validation",
                download=False,
                cache_rate=0.0,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        print("Creating train data loader", flush=True)
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        print("Creating val data loader", flush=True)
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)


class Net(LightningModule):
    def __init__(self, max_epochs=3, upsample_mode="deconv"):
        super().__init__()
        self._model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
            upsample_mode=upsample_mode)
        self.loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True,
                                      to_onehot_y=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.max_epochs = max_epochs
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        self.num_images += batch["image"].shape[0]
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self):
        self.start_time = time.time()
        self.num_images = 0

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.start_time
        throughput = self.num_images / epoch_time
        self.log("images/sec", throughput, on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True, logger=True, reduce_fx="sum")

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        outputs = sliding_window_inference(
            inputs=images,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=self.forward,
            overlap=0.5,
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        self.log("val_dice", mean_val_dice, sync_dist=True)
        self.log("val_loss", mean_val_loss, sync_dist=True)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory


def main(args):
    adapt_monai_to_gaudi()
    brats_dm = BratsDataModule(args.data_dir, batch_size=args.batch_size, num_workers=args.num_data_workers)
    set_determinism(seed=0)

    # initialise the LightningModule
    net = Net()
    tb_logger = TensorBoardLogger("./tb_logs", name="segresnet")

    # Wrap model in HPU graphs
    htcore.hpu.ModuleCacher(max_graphs=10)(model=net, inplace=True)

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_dice",
        mode="max",
        dirpath="./checkpoints",
        filename="segresnet-{epoch:02d}-{val_dice:.2f}",
    )

    # initialise Lightning's trainer.
    trainer = Trainer(
        accelerator=HPUAccelerator(),
        devices=args.num_devices,
        plugins=[
            HPUPrecisionPlugin(
                precision="bf16-mixed",
            )
        ],
        max_epochs=args.train_epochs,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        num_sanity_val_steps=1,
        log_every_n_steps=10,
    )

    # train
    trainer.fit(net, datamodule=brats_dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Brats 3D Segmentation Model')
    parser.add_argument(
        "--num_devices",
        default=8,
        type=int,
        required=True,
        help="Number of HPU devices for training.",
    )
    parser.add_argument(
        "--train_epochs",
        default=300,
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        required=True,
        help="BRATS data directory. Data downloaded if directory does not exist.",
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Train batch size.",
    )
    parser.add_argument(
        "--num_data_workers",
        default=10,
        type=int,
        help="Number of dataloader workers.",
    )

    args = parser.parse_args()
    main(args)
