#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# coding: utf8

import sys

import warnings
import pathlib
import json
import argparse

from PIL import Image
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib

import habana_frameworks.torch.core
import habana_frameworks.torch.distributed.hccl
import habana_frameworks.torch.hpu

import os
os.environ["PT_HPU_LAZY_MODE"] = "1" #run lazy mode
os.environ['PT_HPU_ENABLE_SYNC_OUTPUT_HOST'] = 'false'

CLASS_LIST = [
    "Car", "Truck", "Pedestrian", "Bicycle", "Traffic signal",
    "Traffic sign", "Utility vehicle", "Sidebars", "Speed bumper", "Curbstone",
    "Solid line", "Irrelevant signs", "Road blocks", "Tractor", "Non-drivable street",
    "Zebra crossing", "Obstacles / trash", "Poles", "RD restricted area", "Animals",
    "Grid structure", "Signal corpus", "Drivable cobblestone", "Electronic traffic", "Slow drive area",
    "Nature object", "Parking area", "Sidewalk", "Ego car", "Painted driv. instr.",
    "Traffic guide obj.", "Dashed line", "RD normal street", "Sky", "Buildings",
    "Blurred area", "Rain dirt"
]

def get_label_file(image_file):
    return image_file.parent.parent.parent / 'label' / \
        image_file.parent.name / \
        image_file.name.replace('camera', 'label')

def display_images(images_with_labels):
    fig, ax = plt.subplots(1, len(images_with_labels))
    if len(images_with_labels) == 1:
        ax = [ax]
    for i, (image, label) in enumerate(images_with_labels):
        ax[i].imshow(image)
        ax[i].set_title(label)
        ax[i].set_axis_off()
    plt.show()


def get_masks(label_image, color, class_id):
    preliminary_mask = (label_image[0].eq(
        color[0]) * label_image[1].eq(color[1]) * label_image[2].eq(color[2])) # T/F for each pixel
    
    final_mask = class_id * np.asarray(preliminary_mask).astype(int)

    return final_mask #[1208, 1920] each pixel contains class info

def get_classes(data_location, class_list):
    available_classes_file = data_location / 'class_list.json'
    with open(available_classes_file, 'r') as f:
        available_classes = json.load(f)

    classes = [
        {
            "name": class_name, 
            "id": id+1, 
            "colors": [
                [int(color[i:i+2], 16) for i in (1, 3, 5)]
                for color in available_classes
                if class_name in available_classes[color]
            ]
        }
        for id, class_name in enumerate(class_list)
    ]
    return classes


# Dataset class - responsible for reading files from A2D2 dataset and returning datapoints. Each datapoint consist of an image with respective list of labels, binary masks and bounding boxes.
class A2D2(torch.utils.data.Dataset):
    def __init__(self, data_location, classes, image_transform=None, label_transform=None):
        self.classes = classes
        self.images = [
            image_file
            for image_file in data_location.rglob('*_camera_*.png')
        ]
        self.image_transform = image_transform
        self.label_transform = label_transform

        self.color_id = []
        for class_dict in self.classes:
            for color in class_dict['colors']:
                self.color_id.append((color, class_dict['id']))

    def __getitem__(self, index):
        '''
        index: the index of item
        Return image and its labels
        '''
        return self.get_image(index, self.image_transform), self.get_target(index, self.label_transform)

    def get_image(self, index, image_transform):
        image_file = self.images[index]
        input_image = Image.open(image_file)
        if image_transform:
            input_image = image_transform(input_image)
        return input_image

    def get_target(self, index, label_transform):
        label_file = get_label_file(self.images[index])
        # Load target label file
        label_image = Image.open(label_file)
        if label_transform:
            label_image = label_transform(label_image)

        # create masks
        masks_list = [
            get_masks(label_image, color, class_id)
            for color, class_id in self.color_id
        ]
        masks = np.maximum.reduce(masks_list, keepdims=True)

        return masks


    def __len__(self):
        return len(self.images)

# Data Module class - responsible for creating training and validation dataloaders. Additionally stores definitions of image transformations applied to input data.
from abc import ABC
import habana_dataloader
class A2D2DataModule(ABC):
    def __init__(self, data_location, class_list, batch_size, num_workers):
        super().__init__()

        self.data_location = data_location
        self.classes = get_classes(data_location, class_list)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_visualization_transform = transforms.Compose([
            transforms.Resize((192,160)),
            transforms.ToTensor(),
        ])
        self.image_transform = transforms.Compose([
            self.image_visualization_transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(
                (192,160),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.PILToTensor(),
        ])
        
        full_dataset = A2D2(self.data_location, self.classes,
                            self.image_transform, self.label_transform)
        train_ds_len = int(0.9*len(full_dataset))
        self.train_ds, self.valid_ds = torch.utils.data.random_split(
            full_dataset, (train_ds_len, len(full_dataset)-train_ds_len))
        self.sampler = torch.utils.data.DistributedSampler(self.train_ds)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_ds,
            batch_size=64,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

from models.unet import UNet
from models.loss import Loss
from models.metrics import Dice
from models.nn_unet import NNUnet
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from utils.utils import mark_step, is_main_process

def get_dllogger(results):
    if not os.path.exists(results) and is_main_process():
        os.mkdir(results)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(results, "logs.json")),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
        ]
    )

class NNUnetA2D2(NNUnet):
    def __init__(self, args):
        super(NNUnet, self).__init__()
        self.args = args
        if hasattr(self, 'save_hyperparameters'):
            self.save_hyperparameters()
        self.patch_size = [192,160]
        self.n_class = 37 #excluding background
        self.model = UNet(
            in_channels=3,
            n_class=38, #including background
            kernels=[[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]],
            strides=[[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]],
            dimension=self.args.dim,
            residual=self.args.residual,
            attention=self.args.attention,
            drop_block=self.args.drop_block,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
            deep_supervision=self.args.deep_supervision,
            )

        self.loss = Loss(self.args.focal)
        self.dice = Dice(self.n_class)
        self.best_sum = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.learning_rate = args.learning_rate
        self.tta_flips = [[2], [3], [2, 3]]
        self.test_idx = 0
        self.test_imgs = []
        if self.args.exec_mode in ["train", "evaluate"]:
            self.dllogger = get_dllogger(args.results)

    def configure_optimizers(self):
        self.model = self.model.to(torch.device("hpu"))
        self.model = DDP(self.model, bucket_cap_mb=self.args.bucket_cap_mb,
                            gradient_as_bucket_view=True, static_graph=True)
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW
        optimizer = FusedAdamW(self.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=self.args.weight_decay)
        scheduler = {
            "none": None,
            "multistep": torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.steps, gamma=self.args.factor),
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs),
        }[self.args.scheduler.lower()]
        opt_dict = {"optimizer": optimizer, "monitor": "val_loss"}
        if scheduler is not None:
            opt_dict.update({"lr_scheduler": scheduler})
        return opt_dict
    
    def validation_step(self, batch, batch_idx):
        self.current_epoch = self.trainer.current_epoch
        if self.current_epoch < self.args.skip_first_n_eval:
            return None
        img, lbl = batch[0], batch[1]
        img, lbl = img.to(torch.device("hpu"), non_blocking=True), lbl.to(torch.device("hpu"), non_blocking=True)
        pred = self.forward(img)
        self.dice.update(pred, lbl[:, 0])
        loss = self.loss(pred, lbl)
        mark_step(self.args.run_lazy_mode)
        return {"val_loss": loss}
    
    def get_train_data(self, batch):
        img, lbl = batch[0], batch[1]
        img, lbl = img.to(torch.device("hpu"), non_blocking=True), lbl.to(device=torch.device("hpu"), dtype=torch.float32, non_blocking=True)
        return img, lbl
    
    def forward(self, img):
        return self.model(img)

from pytorch.trainer import Trainer
from pytorch.early_stopping_unet import EarlyStopping

import time

class TrainerA2D2(Trainer):
    def __init__(self, hparams):
        Trainer.__init__(self, hparams)
    
    def save_ckpt(self, ckpt_path, model, num_epoch):
        print(f"Saving weights to {ckpt_path}", end="\r")

        ckpt_state = {
                "start_epoch": num_epoch + 1,
                "model": model.state_dict(),
        }
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        filename = os.path.join(ckpt_path, str(num_epoch) + "_ckpt.pth")
        torch.save(ckpt_state, filename)

    def fit(self, model:"NNUnet", hparams) -> None:
            model.trainer = self
            self.earlystopping =  EarlyStopping(monitor="dice_sum", patience=hparams.args.patience, verbose=True, mode="max")
            self.earlystopping.setup(self)
            self.optimizer = hparams.opt_dict['optimizer']
            self.scheduler = hparams.opt_dict['lr_scheduler'] if 'lr_scheduler' in  hparams.opt_dict else None

            self.train_dataloaders.append(hparams.data_module.train_dataloader())
            self.val_dataloaders.append(hparams.data_module.val_dataloader())
            for self.current_epoch in range(hparams.args.max_epochs):
                hparams.data_module.sampler.set_epoch(self.current_epoch)
                time_1 = time.time()
                train_output_loss = self._implement_train(model, hparams)
                val_output = self._implement_validate(model, hparams)
                self.scheduler_step(val_output['val_loss'])

                if is_main_process():
                    self.save_ckpt(hparams.args.results, model, self.current_epoch)

                    time_2 = time.time()
                    time_interval = time_2 - time_1
                    print(f"End epoch: {self.current_epoch} with time interval: {time_interval:.3f} secs")

                if self.current_epoch >= model.args.min_epochs:
                     self.earlystopping.on_validation_end(self, val_output['dice_sum'])
                     if self.earlystopping.stopped_epoch == self.current_epoch:
                        print(f"Training stopped with epoch: {self.current_epoch}")
                        break

from utils.utils import set_seed, seed_everything

def setup(rank, world_size, hparams):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["ID"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"]=str(rank)
    #distributed package for HCCL
    dist._DEFAULT_FIRST_BUCKET_BYTES = 200*1024*1024  #200MB
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    set_seed(hparams.args.model_seed)
    hparams.model = NNUnetA2D2(hparams.args)

def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size, hparams):
    print(f"Running Unet 8x example on rank {rank}.")
    setup(rank, world_size, hparams)

    model = NNUnetA2D2(hparams.args)
    hparams.opt_dict=model.configure_optimizers()

    set_seed(hparams.args.data_module_seed)
    hparams.data_module = A2D2DataModule(pathlib.Path(hparams.args.data_path), CLASS_LIST, hparams.args.batch_size, hparams.args.num_workers)

    trainer = hparams.trainer
    trainer.fit(model, hparams)

    cleanup()

def run_demo(demo_fn, world_size, hparams):

    mp.spawn(demo_fn,
        args=(world_size,hparams),
        nprocs=world_size, join=True)

if __name__ == "__main__":

    os.environ['framework']='NPT'
    from types import SimpleNamespace
    hparams = SimpleNamespace()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Unet2D training for Autonomous Driving')
    parser.add_argument('--data_path', type=str, help='Data path', default="/root/data/camera_lidar_semantic")
    parser.add_argument('--exec_mode', type=str, choices=["train", "evaluate", "predict"], default="train", help="Execution mode to run the model"),
    parser.add_argument('--results', type=str, help='Output path', default="./results")
    parser.add_argument('--benchmark', action="store_true", help="Run model benchmarking")
    parser.add_argument('--task', type=str, default="01", help="Task number. MSD uses numbers 01-10")
    parser.add_argument('--hpus', type=int, default=8, help="Number of hpus")
    parser.add_argument('--gpus', type=int, default=0, help="Number of gpus")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=11, help="Number of subprocesses to use for data loading")
    parser.add_argument('--norm', type=str, default="instance", help="Normalization layer")
    parser.add_argument('--affinity', type=str, default="disabled", help="type of CPU affinity")
    parser.add_argument('--dim', type=int, default=2, help="UNet dimension")
    parser.add_argument('--optimizer', type=str, default="fusedadamw",help="Optimizer")
    parser.add_argument('--deep_supervision', action="store_true", help="Enable deep supervision")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--residual', action="store_true", help="Enable residual block in encoder")
    parser.add_argument('--attention', action="store_true", help="Enable attention in decoder")
    parser.add_argument('--drop_block', action="store_true", help="Enable drop block")
    parser.add_argument('--negative_slope', type=float, default=0.01, help="Negative slope for LeakyReLU")
    parser.add_argument('--focal', action="store_true", help="Use focal loss instead of cross entropy")
    parser.add_argument('--patience', type=int, default=100, help="Early stopping patience")
    parser.add_argument('--scheduler', type=str, default="None", choices=["none", "multistep", "cosine"], help="Learning rate scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay (L2 penalty)")
    parser.add_argument('--min_epochs', type=int, default=50, help="Force training for at least these many epochs")
    parser.add_argument('--max_epochs', type=int, default=100, help="Stop training after this number of epochs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before collecting statistics")
    parser.add_argument('--gradient_clip', action="store_true", help="Enable gradient_clip to improve training stability")
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=25, help="set progress_bar_refresh_rate")
    parser.add_argument('--run_lazy_mode', action="store_false", help="Use lazy mode")
    parser.add_argument('--skip_first_n_eval', type=int, default=0, help="Skip the evaluation for the first n epochs.")
    parser.add_argument('--val_batch_size', type=int, default=64, help="Validation batch size")
    parser.add_argument("--steps", nargs="+", required=False, help="Steps for multistep scheduler")
    parser.add_argument("--factor", type=float, default=0.3, help="Scheduler factor")
    parser.add_argument('--bucket_cap_mb', type=int, default=125, help="Size in MB for the gradient reduction bucket size")
    parser.add_argument("--is_autocast", action='store_true', required=False, help='Enable autocast')
    args = parser.parse_args()

    hparams.args = args

    seed_everything(seed=args.seed)
    torch.backends.cuda.deterministric = True
    torch.use_deterministic_algorithms(True)
    set_seed(args.seed)

    hparams.args.data_module_seed = np.random.randint(0, 1e6)

    hparams.args.model_seed = np.random.randint(0, 1e6)

    hparams.world_size = hparams.args.hpus
    hparams.hpus = hparams.args.hpus

    trainer_seed = np.random.randint(0, 1e6)

    seed_everything(args.seed)

    set_seed(trainer_seed)
    hparams.trainer = TrainerA2D2(hparams)

    run_demo(demo_basic, hparams.world_size, hparams)
