## 1. Environment Preparation


**Dataset Preparation:**

1. Training: [VGGFace2](https://github.com/NNNNAI/VGGFace2-HQ) and [Asian-Celeb](https://github.com/deepinsight/insightface/issues/1977).
(Since Asian-Celeb seems not available now, you may generate pickle files for your own training dataset by yourself 
following our [image_512_quality.pickle (code:ygss)](https://pan.baidu.com/s/1f8JbTPbj4a243dxgP7Jj7g) pickle 
file format: `List[["img_absolute_path.jpg"(str), iqa_score(float)]]`.)

2. Testing: [FF+](https://github.com/ondyari/FaceForensics),
[CelebA-HQ](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download), 
[FFHQ](https://github.com/NVlabs/ffhq-dataset), 
[Web]()

**Model Dependencies:** [Baiduyun Drive (code:ygss)](https://pan.baidu.com/s/1jtGYuYEhgzTOeXm1Aed_NA)

1. Models used by face swapping baselines (FaceShifter, SimSwap):

- arcface: Copy [weights/third_party/arcface/*]() to `ReliableSwap/modules/third_party/arcface/*`

2. Models for synthesizing naive/cycle triplets:

- lia: See [supervision/readme.md](supervision/readme.md)
- gpen: See [supervision/readme.md](supervision/readme.md)
- deepfillv2: See [supervision/readme.md](supervision/readme.md)
- dml_csr: See [supervision/readme.md](supervision/readme.md)
- faceiqa: See [supervision/readme.md](supervision/readme.md)

3. Third party models for training and testing:

All model weights are included in [weights/third_party (code:ygss)](https://pan.baidu.com/s/1jtGYuYEhgzTOeXm1Aed_NA).
Copy [weights/third_party/XXX/*]() to `ReliableSwap/modules/third_party/XXX/*`.

- vgg: `VGG/vgg19-dcbb9e9d.pth`
- bisenet: `bisenet/79999_iter.pth`
- BFM: `BFM/*`
- cosface*: `cosface/net_sphere20_data_vggface2_acc_9955.pth`
- deep3d(+): `Deep3D/*`
- hopenet(+): `hopenet/hopenet_robust_alpha1.pth`
- pytorch_fid(+): `pytrch_fid/pt_inception-2015-12-05-6726825d.pth`

(`+` means the model is only used for evaluation.)

4. Other third party models (irrelevant to cycle triplets and FixerNet):

These models belong to SOTA face swapping methods.
Copy [weights/third_party/XXX/*]() to `ReliableSwap/modules/third_party/XXX/*`.

- hires: `HiRes/*`
- infoswap: `InfoSwap/*` (Please visit their [website](https://github.com/GGGHSL/InfoSwap-master) to apply for the pre-trained InfoSwap model.)
- megafs: `MegaFS/*`
- simswap (official): `SimSwap/*`
  (*[Note]* Official SimSwap requires copying checkpoints to its root folder.)

**ReliableSwap pre-trained checkpoints and weights:** 

All model `.ckpt` and `.pth` files are included in [reliableswap_weights](https://pan.baidu.com/s/1z6VgH3TA7qCJtw9cqDKJUg).
Copy [reliableswap_weights/extracted_pth/*.pth]() to `ReliableSwap/inference/extracted_pth/`.
The weights extracted from checkpoints and the corresponding checkpoints are listed as follows. 

- FaceShifter (vanilla): `extracted_pth/G_tmp_v5.pth` and `ckpt/faceshifter_vanilla_5/epoch=11-step=548999.ckpt`
- ReliableSwap (w/ FaceShifter): `extracted_pth/G_mouth1_t38_post.pth` and `ckpt/triplet10w_38/epoch=11-step=440999.ckpt`
- SimSwap (vanilla): `extracted_pth/G_tmp_sv4_off.pth` and `ckpt/simswap_vanilla_4/epoch=694-step=1487999.ckpt`
- ReliableSwap (w/ SimSwap): `extracted_pth/G_mouth1_st5.pth` and `ckpt/simswap_triplet_5/epoch=12-step=782999.ckpt`