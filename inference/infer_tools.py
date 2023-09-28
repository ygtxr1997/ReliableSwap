import os

import yaml
import pickle
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl

from inference.gradio_demo.tricks import (
global_trick, vgg_mean, vgg_std, global_bisenet, global_smooth_mask
)
from trainer.faceshifter.faceshifter_pl import FaceshifterPL
from trainer.simswap.simswap_pl import SimSwapPL
from infoswap.inference_model import InfoSwapInference
from hires.image_infer import HiResImageInfer
from megafs.image_infer import MegaFSImageInfer


make_abs_path = lambda fn: os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), fn))


class InferFolderDataset(Dataset):
    def __init__(self, in_folder: str,
                 source_fn: str = "S_cropped.png",
                 target_fn: str = "T_cropped.png",
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)),
                 ]),
                 ):
        self.in_folder = in_folder
        self.source_fn = source_fn
        self.target_fn = target_fn

        self.transform = transform

        pair_folders = os.listdir(in_folder)
        self.pair_folders = pair_folders

    def __getitem__(self, index):
        pair_path = os.path.join(self.in_folder, self.pair_folders[index])
        s_path = os.path.join(pair_path, self.source_fn)
        t_path = os.path.join(pair_path, self.target_fn)

        s_img = np.array(Image.open(s_path).convert("RGB")).astype(np.uint8)
        t_img = np.array(Image.open(t_path).convert("RGB")).astype(np.uint8)

        if self.transform is not None:
            s_img = self.transform(s_img)
            t_img = self.transform(t_img)

        return {
            "source_image": s_img,
            "target_image": t_img,
            "pair_name": self.pair_folders[index],
        }

    def __len__(self):
        return len(self.pair_folders)


class BaseModelInfer(object):
    def load_model(self, in_ckpt: str, out_pt: str, **kwargs) -> any:
        pass

    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _get_fixer_param(use_fixer: str = None):
        if 'mouth1' == use_fixer:
            fixer_net_param = {
                "use": True,
                "feature_dim": 128,
                "crop_param": (28, 56, 84, 112),
                "weight_path": make_abs_path("../pretrained/third_party/arcface/mouth_net_28_56_84_112.pth"),
            }
        elif 'mouth2' == use_fixer:
            fixer_net_param = {
                "use": True,
                "feature_dim": 128,
                "crop_param": (19, 74, 93, 112),
                "weight_path": make_abs_path("../pretrained/third_party/arcface/mouth_net_19_74_93_112.pth"),
            }
        else:
            fixer_net_param = {
                "use": False
            }
        return fixer_net_param


class FaceShifterInfer(BaseModelInfer):
    def __init__(self, in_ckpt: str, out_pt: str,
                 use_fixer: str = None,
                 use_hair_post: bool = False,
                 use_mouth_post: bool = False,
                 ):
        model, fixer_net_param = self.load_model(in_ckpt, out_pt, use_fixer)
        self.model = model
        self.fixer_net_param = fixer_net_param

        self.use_hair_post = use_hair_post
        self.use_mouth_post = use_mouth_post

    def load_model(self, in_ckpt: str, out_pt: str,
                   use_fixer: str = None) -> any:
        fixer_net_param = self._get_fixer_param(use_fixer)
        model = self._extract_pt(in_ckpt, out_pt, fixer_net_param)
        model.eval()
        model = model.cuda()
        return model, fixer_net_param

    @torch.no_grad()
    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor):
        i_r = self.model(i_s, i_t)[0]  # x, id_vector, att
        if self.use_hair_post:
            i_r = global_trick.finetune_hair(i_t, i_r)
        if self.use_mouth_post:
            i_r = global_trick.finetune_mouth(i_s, i_t, i_r)
        return i_r

    @staticmethod
    def _extract_pt(in_ckpt: str, out_pt: str,
                    fixer_net_param: dict = None,
                    config_file: str = '../trainer/faceshifter/config.yaml',
                    ):
        with open(make_abs_path(config_file), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['mouth_net'] = fixer_net_param
        net = FaceshifterPL(config=config)
        ckpt_weight = torch.load(in_ckpt, map_location="cpu")
        net.load_state_dict(ckpt_weight["state_dict"], strict=False)
        net.eval()
        torch.save(net.generator.state_dict(), out_pt)
        return net.generator


class ReliableSwapFaceShifterInfer(FaceShifterInfer):
    def __init__(self, in_ckpt: str, out_pt: str,
                 use_fixer: str = "mouth1",
                 **kwargs):
        super().__init__(in_ckpt, out_pt, use_fixer=use_fixer, **kwargs)


class SimSwapInfer(BaseModelInfer):
    def __init__(self, in_ckpt: str, out_pt: str,
                 use_official_arc: bool = False,
                 use_fixer: str = None,
                 use_hair_post: bool = False,
                 use_mouth_post: bool = False,
                 ):
        self.use_official_arc = use_official_arc
        self.use_hair_post = use_hair_post
        self.use_mouth_post = use_mouth_post

        model, net_arc, fixer_net, fixer_net_param = self.load_model(in_ckpt, out_pt, use_fixer)
        self.model = model
        self.net_arc = net_arc
        self.fixer_net = fixer_net
        self.fixer_net_param = fixer_net_param

    def load_model(self, in_ckpt: str, out_pt: str,
                   use_fixer: str = None) -> any:
        fixer_net_param = self._get_fixer_param(use_fixer)
        model, net_arc, fixer_net = self._extract_pt(in_ckpt, out_pt, fixer_net_param,
                                                     use_official_arc=self.use_official_arc)
        model.eval()
        model = model.cuda()
        return model, net_arc, fixer_net, fixer_net_param

    @torch.no_grad()
    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor):
        i_r = self.model(source=i_s, target=i_t, net_arc=self.net_arc, mouth_net=self.fixer_net)
        if self.use_hair_post:
            i_r = global_trick.finetune_hair(i_t, i_r)
        if self.use_mouth_post:
            i_r = global_trick.finetune_mouth(i_s, i_t, i_r)
        return i_r.clamp(-1, 1)

    @staticmethod
    def _extract_pt(in_ckpt: str, out_pt: str,
                    fixer_net_param: dict = None,
                    config_file: str = '../trainer/simswap/config.yaml',
                    use_official_arc: bool = False,
                    ):
        with open(make_abs_path(config_file), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['mouth_net'] = fixer_net_param
        net = SimSwapPL(config=config, use_official_arc=use_official_arc)
        ckpt_weight = torch.load(in_ckpt, map_location="cpu")
        net.load_state_dict(ckpt_weight["state_dict"], strict=False)
        net.eval()
        torch.save(net.netG.state_dict(), out_pt)
        return net.netG, net.netArc, net.mouth_net


class ReliableSwapSimSwapInfer(SimSwapInfer):
    def __init__(self, in_ckpt: str, out_pt: str,
                 use_fixer: str = "mouth1",
                 **kwargs):
        super().__init__(in_ckpt, out_pt, use_fixer=use_fixer, **kwargs)


class InfoSwapInfer(BaseModelInfer):
    def __init__(self, **kwargs):
        self.model = None
        self.load_model("", "")

    def load_model(self, in_ckpt: str, out_pt: str, **kwargs) -> any:
        self.model = InfoSwapInference()
        self.model.load_model()

    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor) -> torch.Tensor:
        i_r = self.model.infer_batch(i_s, i_t)
        i_r = i_r.cuda()
        i_r = F.interpolate(i_r, size=(256, 256), mode='bilinear', align_corners=True)
        return i_r


class HiResInfer(BaseModelInfer):
    def __init__(self, **kwargs):
        self.model = None
        self.load_model("", "")

    def load_model(self, in_ckpt: str, out_pt: str, **kwargs) -> any:
        self.model = HiResImageInfer()

    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor) -> torch.Tensor:
        from PIL import Image
        i_s = (i_s + 1) * 127.5
        i_t = (i_t + 1) * 127.5
        source_np = i_s.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.uint8)
        target_np = i_t.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.uint8)
        source_pil = Image.fromarray(source_np)
        target_pil = Image.fromarray(target_np)
        i_r = self.model.image_infer(source_pil=source_pil,
                                     target_pil=target_pil)
        i_r = F.interpolate(i_r, size=256, mode="bilinear", align_corners=True)
        i_r = i_r.clamp(-1, 1)
        return i_r


class MegaFSInfer(BaseModelInfer):
    def __init__(self, **kwargs):
        self.model = None
        self.load_model("", "")

    def load_model(self, in_ckpt: str, out_pt: str, **kwargs) -> any:
        self.model = MegaFSImageInfer()

    def infer_batch(self, i_s: torch.Tensor, i_t: torch.Tensor) -> torch.Tensor:
        i_s = F.interpolate(i_s, size=256, mode="bilinear", align_corners=True)
        i_t = F.interpolate(i_t, size=256, mode="bilinear", align_corners=True)
        i_r = self.model.image_infer(source_tensor=i_s,
                                     target_tensor=i_t)
        if torch.isnan(i_r).any():
            print('NAN in i_r, will be set to 0.')
            i_r = torch.where(torch.isnan(i_r), torch.full_like(i_r, 0.), i_r)
        i_r = i_r.clamp(-1, 1)
        return i_r


class TestIterator(pl.LightningModule):
    def __init__(self, in_folder: str,
                 out_folder: str,
                 model_infer: BaseModelInfer,
                 batch_size: int = 1,
                 use_gpen: bool = False,
                 ):
        super().__init__()
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.model_infer = model_infer
        self.batch_size = batch_size
        self.use_gpen = use_gpen

        self.test_dataset = InferFolderDataset(
            in_folder
        )
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(os.path.join(out_folder, "source"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "target"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "result"), exist_ok=True)

        self.t_imgs = []
        self.s_imgs = []
        self.r_imgs = []

        print(f"[TestIterator] successfully loaded model: {type(model_infer)}")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def forward(self, source_img: torch.Tensor, target_img: torch.Tensor):
        return self.model_infer.infer_batch(source_img, target_img)

    def test_step(self, batch, batch_idx):
        i_t = batch["target_image"]
        i_s = batch["source_image"]
        pair_name = batch["pair_name"]

        i_r = self.forward(i_s, i_t)  # forward, (B,C,H,W)

        self._save_img(i_t, i_s, i_r, batch_idx, pair_name)

    def _save_img(self, i_t, i_s, i_r, batch_idx, pair_name: str):
        img_t = self._nchw_to_1hwrgb(i_t)
        img_s = self._nchw_to_1hwrgb(i_s)
        img_r = self._nchw_to_1hwrgb(i_r)

        img_r = global_trick.gpen(img_r, self.use_gpen)

        img_t = Image.fromarray(img_t)
        img_s = Image.fromarray(img_s)
        img_r = Image.fromarray(img_r)

        save_name = "%05d.jpg" % batch_idx
        img_t.save(os.path.join(self.out_folder, "target", save_name))
        img_s.save(os.path.join(self.out_folder, "source", save_name))
        img_r.save(os.path.join(self.out_folder, "result", save_name))

        self.t_imgs.append(img_t)
        self.r_imgs.append(img_r)
        self.s_imgs.append(img_s)

    @staticmethod
    def _nchw_to_1hwrgb(tensor, b_idx: int = 0):
        return ((tensor + 1.) * 127.5).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[b_idx]


def get_test_interator(model_name: str,
                       in_folder: str,
                       out_folder: str,
                       batch_size: int = 1,
                       use_gpen: bool = False,
                       ):
    if model_name == "faceshifter":
        infer_model = FaceShifterInfer(
            "../pretrained/reliableswap_weights/ckpt/faceshifter_vanilla_5/epoch=11-step=548999.ckpt",
            "../pretrained/reliableswap_weights/extracted_pth/faceshifter_v5.pt"
        )
    elif model_name == "reliable_faceshifter":
        infer_model = FaceShifterInfer(
            "../pretrained/reliableswap_weights/ckpt/triplet10w_38/epoch=11-step=440999.ckpt",
            "../pretrained/reliableswap_weights/extracted_pth/reliable_faceshifter_38.pt",
            use_fixer="mouth1", use_hair_post=True, use_mouth_post=True
        )
    elif model_name == "simswap":
        infer_model = SimSwapInfer(
            "../pretrained/reliableswap_weights/ckpt/simswap_vanilla_4/epoch=694-step=1487999.ckpt",
            "../pretrained/reliableswap_weights/extracted_pth/simswap_v5.pt",
            use_official_arc=True,
        )
    elif model_name == "reliable_simswap":
        infer_model = ReliableSwapSimSwapInfer(
            "../pretrained/reliableswap_weights/ckpt/simswap_triplet_5/epoch=12-step=782999.ckpt",
            "../pretrained/reliableswap_weights/extracted_pth/reliable_simswap_5.pt",
            use_fixer="mouth1", use_hair_post=False, use_mouth_post=False
        )
    elif model_name == "infoswap":
        infer_model = InfoSwapInfer()
    elif model_name == "hires":
        infer_model = HiResInfer()
    elif model_name == "megafs":
        infer_model = MegaFSInfer()
    else:
        raise ValueError("Not supported model_name!")

    test_iterator = TestIterator(
        in_folder, os.path.join(out_folder, model_name),
        model_infer=infer_model,
        batch_size=batch_size, use_gpen=use_gpen
    )
    return test_iterator


if __name__ == "__main__":
    faceshifter_infer = FaceShifterInfer(
        "../pretrained/reliableswap_weights/ckpt/faceshifter_vanilla_5/epoch=11-step=548999.ckpt",
        "../pretrained/reliableswap_weights/extracted_pth/faceshifter_v5.pt"
    )
    reliable_faceshifter_infer = FaceShifterInfer(
        "../pretrained/reliableswap_weights/ckpt/triplet10w_38/epoch=11-step=440999.ckpt",
        "../pretrained/reliableswap_weights/extracted_pth/reliable_faceshifter_38.pt",
        use_fixer="mouth1", use_hair_post=True, use_mouth_post=True
    )
    simswap_infer = SimSwapInfer(
        "../pretrained/reliableswap_weights/ckpt/simswap_vanilla_4/epoch=694-step=1487999.ckpt",
        "../pretrained/reliableswap_weights/extracted_pth/simswap_v5.pt"
    )
    reliable_simswap_infer = ReliableSwapSimSwapInfer(
        "../pretrained/reliableswap_weights/ckpt/simswap_triplet_5/epoch=12-step=782999.ckpt",
        "../pretrained/reliableswap_weights/extracted_pth/reliable_simswap_5.pt",
        use_fixer="mouth1", use_hair_post=False, use_mouth_post=False
    )
    infoswap_infer = InfoSwapInfer()
    hires_infer = HiResInfer()
    megafs_infer = MegaFSInfer()


