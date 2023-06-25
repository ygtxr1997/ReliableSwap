# High-resolution Face Swapping via Latent Semantics Disentanglement
>We present a novel high-resolution face swapping method using the inherent prior knowledge of a pre-trained GAN model. Although previous research can leverage generative priors to produce high-resolution results, their quality can suffer from the entangled semantics of  the latent space. We explicitly disentangle the latent semantics by utilizing the progressive nature of the generator, deriving structure attributes from the shallow layers and appearance attributes from the deeper ones. Identity and pose information within the structure attributes are further separated by introducing a landmark-driven structure transfer latent direction. The disentangled latent code produces rich generative features that incorporate feature blending to produce a plausible swapping result. We further extend our method to video face swapping by enforcing two spatio-temporal constraints on the latent space and the image space. Extensive experiments demonstrate that the proposed method outperforms state-of-the-art image/video face swapping methods in terms of hallucination quality and consistency.
## Description

We present the inference code on CelebA-HQ dataset, Anaconda is required.


**Build the environment**
```
conda env create -f stylegan.yaml
```

**Download the dataset and checkpoint**
1. We have inverted the CelebA-HQ dataset using [pSp](https://github.com/eladrich/pixel2style2pixel) encoder. The whole dataset can be download from this [link](https://drive.google.com/file/d/1TRLvURZpx5xtEnxBXeaaZs1RbReWftBv/view?usp=sharing). After download the dataset, unzip it and move it to the upper folder.
2. The checkpoint can be downloaded from this [link](https://drive.google.com/file/d/1LH4RlxaPnrHAiWEDm3LDp5Sz9H02bzXU/view?usp=sharing). Download it and put it at the current folder.

**Run the inference code**
```
bash run.sh
```
$\color{#FF0000}{Note:}$ 
1. For testing your own image, you need to invert the image to the StyleGAN's latent space using [pSp](https://github.com/eladrich/pixel2style2pixel) for getting the corresponding latent code, and get the face parsing using [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch), and get the landmark using [H3R](https://github.com/baoshengyu/H3R).
2. Make sure that the dataset folder shares the same structure as our CelebA-HQ dataset, or you can modify the `dataset.py` as you wish.




## Citation
```
@inproceedings{xu2022high,
  title={High-resolution face swapping via latent semantics disentanglement},
  author={Xu, Yangyang and Deng, Bailin and Wang, Junle and Jing, Yanqing and Pan, Jia and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7642--7651},
  year={2022}

```
