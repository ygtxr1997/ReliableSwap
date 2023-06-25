import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np

from celeba.dataloader import CelebADataset
from inference.celebahq.dataloader import CelebaHQEvalDataset
from inference.ffhq.dataloader import FFHQEvalDataset
from supervision.dataset.dataloader import BatchTrainDataset


class GlassesDetector(nn.Module):
    def __init__(self):
        super(GlassesDetector, self).__init__()
        dim_feature = 128
        self.backbone = resnet18(
            pretrained=False,
            num_classes=dim_feature,
        )
        self.drop = nn.Dropout(0.4)
        self.features = nn.BatchNorm1d(dim_feature, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        self.head = nn.Linear(dim_feature, 2)
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        pred = self.backbone(img)
        pred = self.drop(pred)
        pred = self.features(pred)
        pred = self.head(pred)
        pred = self.softmax(pred)
        return pred


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class GlassesDetectorPL(pl.LightningModule):
    def __init__(self,
                 batch_size: int = 128,
                 ):
        super(GlassesDetectorPL, self).__init__()
        self.batch_size = batch_size
        self.warmup_epoch = 0

        self.model = GlassesDetector()
        self.loss = focal_loss(num_classes=2)

        self.record = {
            'img': [],
            'pred': []
        }

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, anno = batch
        pred = self(img)
        anno = anno.squeeze()

        loss = self.loss(pred, anno)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, anno = batch
        pred = self(img)  # (B,2), in [0.,1.]
        anno = anno.squeeze()  # (B,), in {0,1}

        loss = self.loss(pred, anno)

        self.log("val_loss", loss)
        return pred, anno

    def validation_epoch_end(self, outputs):
        true_cnt, false_cnt, total_cnt = 0, 0, 0
        for idx, output in enumerate(outputs):
            pred, anno = output
            pred = pred.argmax(dim=1)
            diff = torch.abs(pred - anno)

            true_cnt += (1 - diff).sum()
            false_cnt += diff.sum()
            total_cnt += diff.shape[0]
        print('true:%d, false:%d, total:%d, acc:%.2f%%' % (
            true_cnt, false_cnt, total_cnt,
            (true_cnt / total_cnt) * 100
        ))

    def test_step(self, batch, batch_idx):
        t_img = batch["target_image"]  # (B,C,H,W)
        s_img = batch["source_image"]
        pred = self(t_img)  # (B,2), in [0.,1.], 0:no occ, 1:occ

        B = t_img.shape[0]
        for b in range(B):
            self.record['img'].append(t_img[b])
            self.record['pred'].append(pred[b][1])

    def test_epoch_end(self, outputs):
        import os
        from PIL import Image
        from tqdm import tqdm
        out_folder = 'ffhq'
        os.system('rm -r %s' % out_folder)
        os.makedirs(out_folder, exist_ok=True)
        total_len = len(self.record['img'])
        print('total imgs: %d' % total_len)
        for idx in tqdm(range(total_len)):
            img = self.record['img'][idx]
            img = ((img + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img = Image.fromarray(img)

            pred = self.record['pred'][idx]
            pred = int(pred * 10000)

            save_name = '%d_%d.jpg' % (pred, idx)
            img.save(os.path.join(out_folder, save_name))

    def configure_optimizers(self):
        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < self.warmup_epoch else 0.1 ** len(
                [m for m in [3, 6, 9] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_step_func
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def train_dataloader(self):
        dataset = CelebADataset(
            celeba_folder='/gavin/datasets/celeba/ffhq_aligned/',
            split_type='train')
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=32,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = CelebADataset(
            celeba_folder='/gavin/datasets/celeba/ffhq_aligned/',
            split_type='val')
        return DataLoader(dataset, batch_size=32, num_workers=8)

    def test_dataloader(self):
        celebahq_ts_list = np.arange(20000)
        np.random.shuffle(celebahq_ts_list)
        dataset = CelebaHQEvalDataset(
            ts_list=celebahq_ts_list,
            dataset_len=1000,
        )
        # dataset = BatchTrainDataset(
        #     top_k=100,
        # )
        return DataLoader(dataset, batch_size=32, num_workers=8)


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import transforms
    path = '../../../supervision/demo_snapshot/demo_17/source.jpg'
    img = Image.open(path, 'r').convert('RGB')
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])
    x_in = trans(img)
    x_in = x_in[None, :, :, :]

    pl_model = GlassesDetectorPL.load_from_checkpoint(
        checkpoint_path='out/tmp/epoch=25-step=16499.ckpt',
    ).eval()
    y_out = pl_model(x_in)
    y_out = y_out[0]
    print('glasses probability = %.2f' % (y_out[1] * 100))
