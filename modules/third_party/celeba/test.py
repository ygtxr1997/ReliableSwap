import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger

from celeba.model import GlassesDetectorPL


parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--gpus",
    type=str,
    default=None,
    help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.",
)
parser.add_argument("-pj", "--project", type=str, default="OccDetector", help="Name of the project.")
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the run.")


parser.add_argument(
    "-rp",
    "--resume_checkpoint_path",
    type=str,
    default='out/tmp/epoch=24-step=15499.ckpt',
    help="path of checkpoint for resuming",
)
parser.add_argument(
    "-p",
    "--checkpoint_path",
    type=str,
    default="out",
    help="saving folder",
)
parser.add_argument(
    "--wandb_resume",
    type=str,
    default=None,
    help="resume wandb logging from the input id",
)

parser.add_argument("-bs", "--batch_size", type=int, default=128, help="bs.")
parser.add_argument("-fs", "--fast_dev_run", type=bool, default=False, help="pytorch.lightning fast_dev_run")
args = parser.parse_args()

pl_model = GlassesDetectorPL.load_from_checkpoint(
    checkpoint_path=args.resume_checkpoint_path,
)

trainer = pl.Trainer(
    gpus=1,
    resume_from_checkpoint=args.resume_checkpoint_path,
    fast_dev_run=args.fast_dev_run,
    progress_bar_refresh_rate=1,
    distributed_backend="dp",
    # strategy="ddp",
    benchmark=True,
)
trainer.test(pl_model)
