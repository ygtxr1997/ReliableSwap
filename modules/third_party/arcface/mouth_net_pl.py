import os.path

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist

from modules.third_party.arcface.mouth_net import MouthNet
from modules.third_party.arcface.margin_loss import Softmax, AMArcFace, AMCosFace
from modules.third_party.arcface.load_dataset import MXFaceDataset, EvalDataset
from modules.third_party.bisenet.bisenet import BiSeNet


class MouthNetPL(pl.LightningModule):
    def __init__(
            self,
            num_classes: int,
            batch_size: int = 256,
            dim_feature: int = 128,
            header_type: str = 'AMArcFace',
            header_params: tuple = (64.0, 0.5, 0.0, 0.0),  # (s, m, a, k)
            rec_folder: str = "/gavin/datasets/msml/ms1m-retinaface",
            learning_rate: int = 0.1,
            crop: tuple = (0, 0, 112, 112),  # (w1,h1,w2,h2)
    ):
        super(MouthNetPL, self).__init__()

        # self.img_size = (112, 112)

        ''' mouth feature extractor '''
        bisenet = BiSeNet(19)
        bisenet.load_state_dict(
            torch.load(
                "/gavin/datasets/hanbang/79999_iter.pth",
                map_location="cpu",
            )
        )
        bisenet.eval()
        bisenet.requires_grad_(False)
        self.mouth_net = MouthNet(
            bisenet=None,
            feature_dim=dim_feature,
            crop_param=crop,
            iresnet_pretrained=False,
        )

        ''' head & loss '''
        self.automatic_optimization = False
        self.dim_feature = dim_feature
        self.num_classes = num_classes
        self._prepare_header(header_type, header_params)
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        ''' dataset '''
        assert os.path.exists(rec_folder)
        self.rec_folder = rec_folder
        self.batch_size = batch_size
        self.crop_param = crop

        ''' validation '''

    def _prepare_header(self, head_type, header_params):
        dim_in = self.dim_feature
        dim_out = self.num_classes

        """ Get hyper-params of header """
        s, m, a, k = header_params

        """ Choose the header """
        if 'Softmax' in head_type:
            self.classification = Softmax(dim_in, dim_out, device_id=None)
        elif 'AMCosFace' in head_type:
            self.classification = AMCosFace(dim_in, dim_out,
                                            device_id=None,
                                            s=s, m=m,
                                            a=a, k=k,
                                            )
        elif 'AMArcFace' in head_type:
            self.classification = AMArcFace(dim_in, dim_out,
                                            device_id=None,
                                            s=s, m=m,
                                            a=a, k=k,
                                            )
        else:
            raise ValueError('Header type error!')

    def forward(self, x, label=None):
        feat = self.mouth_net(x)
        if self.training:
            assert label is not None
            cls = self.classification(feat, label)
            return feat, cls
        else:
            return feat

    def training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)
        img, label = batch

        mouth_feat, final_cls = self(img, label)

        cls_loss = self.cls_criterion(final_cls, label)

        opt.zero_grad()
        self.manual_backward(cls_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5, norm_type=2)
        opt.step()

        ''' loss logging '''
        self.logging_dict({"cls_loss": cls_loss}, prefix="train / ")
        self.logging_lr()
        if batch_idx % 50 == 0 and self.local_rank == 0:
            print('loss=', cls_loss)

        return cls_loss

    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

        lr = -1
        opts = self.trainer.optimizers
        for opt in opts:
            for param_group in opt.param_groups:
                lr = param_group["lr"]
                break
        print('learning rate changed to %.6f' % lr)

    # def validation_step(self, batch, batch_idx):
    #     return self.test_step(batch, batch_idx)
    #
    # def validation_step_end(self, outputs):
    #     return self.test_step_end(outputs)
    #
    # def validation_epoch_end(self, outputs):
    #     return self.test_step_end(outputs)

    @staticmethod
    def save_tensor(tensor: torch.Tensor, path: str, b_idx: int = 0):
        tensor = (tensor + 1.) * 127.5
        img = tensor.permute(0, 2, 3, 1)[b_idx].cpu().numpy()
        from PIL import Image
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(path)

    def test_step(self, batch, batch_idx):
        img1, img2, same = batch
        feat1 = self.mouth_net(img1)
        feat2 = self.mouth_net(img2)
        return feat1, feat2, same

    def test_step_end(self, outputs):
        feat1, feat2, same = outputs
        feat1 = feat1.cpu().numpy()
        feat2 = feat2.cpu().numpy()
        same = same.cpu().numpy()

        feat1 = sklearn.preprocessing.normalize(feat1)
        feat2 = sklearn.preprocessing.normalize(feat2)

        predict_label = []
        num = feat1.shape[0]
        for i in range(num):
            dis_cos = cdist(feat1[i, None], feat2[i, None], metric='cosine')
            predict_label.append(dis_cos[0, 0])
        predict_label = np.array(predict_label)

        return {
            "pred": predict_label,
            "gt": same,
        }

    def test_epoch_end(self, outputs):
        print(outputs)
        pred, same = None, None
        for batch_output in outputs:
            if pred is None and same is None:
                pred = batch_output["pred"]
                same = batch_output["gt"]
            else:
                pred = np.concatenate([pred, batch_output["pred"]])
                same = np.concatenate([same, batch_output["gt"]])
        print(pred.shape, same.shape)

        fpr, tpr, threshold = roc_curve(same, pred)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # choose proper threshold
        print("=> verification finished, acc=%.4f" % (acc))

        ''' save pth '''
        pth_path = "./weights/fixer_net_casia_%s.pth" % ('_'.join((str(x) for x in self.crop_param)))
        self.mouth_net.save_backbone(pth_path)
        print("=> model save to %s" % pth_path)
        mouth_net = MouthNet(
            bisenet=None,
            feature_dim=self.dim_feature,
            crop_param=self.crop_param
        )
        mouth_net.load_backbone(pth_path)
        print("=> MouthNet pth checked")

        return acc

    def logging_dict(self, log_dict, prefix=None):
        for key, val in log_dict.items():
            if prefix is not None:
                key = prefix + key
            self.log(key, val)

    def logging_lr(self):
        opts = self.trainer.optimizers
        for idx, opt in enumerate(opts):
            lr = None
            for param_group in opt.param_groups:
                lr = param_group["lr"]
                break
            self.log(f"lr_{idx}", lr)

    def configure_optimizers(self):
        params = list(self.parameters())
        learning_rate = self.learning_rate / 512 * self.batch_size * torch.cuda.device_count()
        optimizer = torch.optim.SGD(params, lr=learning_rate,
            momentum=0.9, weight_decay=5e-4)
        print('lr is set as %.5f due to the global batch_size %d' % (learning_rate,
                                                                     self.batch_size * torch.cuda.device_count()))

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < 0 else 0.1 ** len(
                [m for m in [11, 17, 22] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001
        scheduler= torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_step_func)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = MXFaceDataset(
            root_dir=self.rec_folder,
            crop_param=self.crop_param,
        )
        train_loader = DataLoader(
            dataset, self.batch_size, num_workers=24, shuffle=True, drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        dataset = EvalDataset(
            rec_folder=self.rec_folder,
            target='lfw',
            crop_param=self.crop_param
        )
        test_loader = DataLoader(
            dataset, 20, num_workers=12, shuffle=False, drop_last=False
        )
        return test_loader


def start_train():
    import os
    import argparse
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        default=None,
        help="Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.",
    )
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the run.")
    parser.add_argument("-pj", "--project", type=str, default="mouthnet", help="Name of the project.")

    parser.add_argument("-rp", "--resume_checkpoint_path",
                        type=str, default=None, help="path of checkpoint for resuming", )
    parser.add_argument("-p", "--saving_folder",
                        type=str, default="/apdcephfs/share_1290939/gavinyuan/out", help="saving folder", )
    parser.add_argument("--wandb_resume",
                        type=str, default=None, help="resume wandb logging from the input id", )

    parser.add_argument("--header_type", type=str, default="AMArcFace", help="loss type.")

    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="bs.")
    parser.add_argument("-fs", "--fast_dev_run", type=bool, default=False, help="pytorch.lightning fast_dev_run")
    args = parser.parse_args()
    args.val_targets = []
    # args.rec_folder = "/gavin/datasets/msml/ms1m-retinaface"
    # num_classes = 93431
    args.rec_folder = "/gavin/datasets/msml/casia"
    num_classes = 10572

    save_path = os.path.join(args.saving_folder, args.name)
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor="train / cls_loss",
        save_top_k=10,
        verbose=True,
        every_n_train_steps=200,
    )

    torch.cuda.empty_cache()
    mouth_net = MouthNetPL(
        num_classes=num_classes,
        batch_size=args.batch_size,
        dim_feature=128,
        rec_folder=args.rec_folder,
        header_type=args.header_type,
        crop=(28, 56, 84, 112)
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
    )

    trainer = pl.Trainer(
        gpus=-1 if args.gpus is None else torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
        logger=logger,
        weights_save_path=save_path,
        resume_from_checkpoint=args.resume_checkpoint_path,
        gradient_clip_val=0,
        max_epochs=25,
        num_sanity_val_steps=1,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=50,
        progress_bar_refresh_rate=1,
        distributed_backend="ddp",
        benchmark=True,
    )
    trainer.fit(mouth_net)


if __name__ == "__main__":

    start_train()
