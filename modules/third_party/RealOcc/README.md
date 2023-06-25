
# Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets
<img src="./resources/teaser.jpg" width="1024">


This repository provides the dataset and code for the following paper:

**Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets**  
Kenny T. R. Voo, [Liming Jiang](https://liming-jiang.com/) and [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)  
In CVPRW 2022.  
[**Paper**](https://arxiv.org/abs/2205.06218) | [**Dataset**](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing) 

> **Abstract:** *This paper performs comprehensive analysis on datasets for occlusion-aware face segmentation, a task that is crucial for many downstream applications. The collection and annotation of such datasets are time-consuming and labor-intensive. Although some efforts have been made in synthetic data generation, the naturalistic aspect of data remains less explored. In our study, we propose two occlusion generation techniques, Naturalistic Occlusion Generation (NatOcc), for producing high-quality naturalistic synthetic occluded faces; and Random Occlusion Generation (RandOcc), a more general synthetic occluded data generation method. We empirically show the effectiveness and robustness of both methods, even for unseen occlusions. To facilitate model evaluation, we present two high-resolution real-world occluded face datasets with fine-grained annotations, RealOcc and RealOcc-Wild, featuring both careful alignment preprocessing and an in-the-wild setting for robustness test. We further conduct a comprehensive analysis on a newly introduced segmentation benchmark, offering insights for future exploration.*

## Updates
- [06/2022] The code of NatOcc and RandOcc are released.
- [04/2022] The RealOcc and RealOcc-Wild datasets are released. 
- [04/2022] This paper is accepted by **CVPRW 2022**.


## Installation
**Clone this repo:**
```bash
git clone https://github.com/kennyvoo/face-occlusion-generation.git
cd face-occlusion-generation/
```
**Dependencies:**

All dependencies for defining the environment are provided in `requirements.txt`.  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/):
```bash
conda create -n occ-gen python=3.8
conda activate occ-gen
pip install -r requirements.txt
pip install -U git+https://github.com/albumentations-team/albumentations
#install pytorch before the following line
conda install -c conda-forge cupy
```
Please install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/).
Please install PyTorch before install cupy
  
## Dataset Preparation

Please download the masks from this [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing) and the images from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), [11k Hands](https://sites.google.com/view/11khands) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/). 

The extracted and upsampled COCO objects images and masks can be found in this [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing).

Please extract CelebAMask-HQ and 11k Hands images based on the splits found in [drive](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5?usp=sharing). 

**Dataset Organization:**

```none

├── dataset
│   ├── CelebAMask-HQ-WO-Train_img
│   │   ├── {image}.jpg
|   ├── CelebAMask-HQ-WO-Train_mask
│   │   ├── {mask}.png
|   ├── DTD
│   │   ├── images
│   │   │   ├── {classA}
│   │   │   │   ├── {image}.jpg
│   │   │   ├── {classB}
│   │   │   │   ├── {image}.jpg
|   ├── 11k-hands_img
│   │   ├── {image}.jpg
|   ├── 11k-hands_mask
│   │   ├── {mask}.png
|   ├── object_image_sr
│   │   ├── {image}.jpg
|   ├── object_image_x4
│   │   ├── {mask}.png

```

## Data Generation

Example script to generate NatOcc dataset 
```bash
bash NatOcc.sh
```
 
Example script to generate RandOcc dataset 
```bash
bash RandOcc.sh
```
Please modify/create the configs file in `/configs` accordingly.

* `RANDOCC` : Choose between NatOcc or RandOcc. If False, NatOcc mode. 
* `SOT` : Specifies whether to use colour transfer via Sliced Optimal Transport
* `ROTATE_AROUND_CENTER` : Rotate hands such that fingers always pointing toward the face.
* `OCCLUSION_MASK` : If True, mask of occlussion will be saved. (For RainNet)

## Result
**Dataset Definition**  
<img src="./resources/dataset_definition.png" height="250">

**Qualitative Result**
![qualitative](./resources/baseline_comparison.jpg)

**Quantitative Result**

![quantitative](./resources/quantitative_result.png)


## Todo
- [X] code release.
- [ ] code optimization.

## Citation
If you find this work useful for your research, please consider citing our paper:  
```bibtex
@inproceedings{voo2022delving,
  title={Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets},
  author={Voo, Kenny T. R. and Jiang, Liming and Loy, Chen Change},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2022}
}
