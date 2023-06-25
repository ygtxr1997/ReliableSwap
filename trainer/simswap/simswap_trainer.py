import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger

from simswap_pl import SimSwapPL


parser = argparse.ArgumentParser()
parser.add_argument(
    "-g",
    "--gpus",
    type=str,
    default=None,
    help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.",
)
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the run.")
parser.add_argument("-pj", "--project", type=str, default="simswap256", help="Name of the project.")

parser.add_argument("-rp", "--resume_checkpoint_path",
    type=str, default=None, help="path of checkpoint for resuming",)
parser.add_argument("-p", "--saving_folder",
    type=str, default="/apdcephfs/share_1290939/gavinyuan/out",help="saving folder",)
parser.add_argument("--wandb_resume",
    type=str, default=None, help="resume wandb logging from the input id",)

parser.add_argument("--config", type=str, default='./config.yaml', help='Path to config yaml file')
parser.add_argument("-sr", "--same_rate", type=int, default=0, help="Same ratio of training pairs.")

parser.add_argument("-bs", "--batch_size", type=int, default=4, help="bs.")
parser.add_argument("-fs", "--fast_dev_run", type=bool, default=False, help="pytorch.lightning fast_dev_run")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

save_path = os.path.join(args.saving_folder, args.name)
os.makedirs(save_path, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    dirpath=save_path,
    monitor="validation / g_loss",
    save_top_k=10,
    verbose=True,
    every_n_train_steps=3000,
)

torch.cuda.empty_cache()
simswap = SimSwapPL(
    batch_size=args.batch_size,
    same_rate=args.same_rate,
    config=config
)

if args.wandb_resume == None:
    resume = "allow"
    wandb_id = wandb.util.generate_id()
else:
    resume = True
    wandb_id = args.wandb_resume
logger = WandbLogger(
    project=args.project,
    entity="gavinyuan",
    name=args.name,
    resume=resume,
    id=wandb_id,
    config=config,
)

trainer = pl.Trainer(
    gpus=-1 if args.gpus is None else torch.cuda.device_count(),
    callbacks=[checkpoint_callback],
    logger=logger,
    weights_save_path=save_path,
    resume_from_checkpoint=args.resume_checkpoint_path,
    gradient_clip_val=0,
    max_epochs=1000,
    num_sanity_val_steps=1,
    fast_dev_run=args.fast_dev_run,
    val_check_interval=1500,
    progress_bar_refresh_rate=1,
    distributed_backend="ddp",
    benchmark=True,
)
trainer.fit(simswap)
