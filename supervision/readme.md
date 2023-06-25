# Triplet Rotational Group Synthesizer

## Usage

### Step 1. Prepare GPU environment

The modules `supervision/parsing/inplace_abn` and `supervison/restoration/neurralgym` needs to be setup
and installed first.
If you are using our docker images: [mirrors.tencent.com/gavinyuan/faceswap:[tag]](),
the subsequent `bash ./env_setup.sh` can be skipped.

On NVIDIA Tesla GPU architecture (e.g. Tesla V100), run:
```shell
bash ./env_setup.sh 7.0
```

On NVIDIA Turing GPU architecture (e.g. Tesla T4), run:
```shell
bash ./env_setup.sh 7.5
```

On NVIDIA Ampere GPU architecture (e.g. A100), run:
```shell
bash ./env_setup.sh 8.0
```

The shell script `env_setup.sh` will run `../tools/env_check.py` to check if the dependency modules are
successfully installed.
```shell
$> [GPU environment OK!]
```

### Step 2. Download pretrained weights

All weights files can be found in [https://drive.weixin.qq.com/s?k=AJEAIQdfAAoGtZsqxL](
https://drive.weixin.qq.com/s?k=AJEAIQdfAAoGtZsqxL).

#### 2.1 LIA (from [LIA GitHub](https://wyhsirius.github.io/LIA-project/))

```shell
cd ./lia_weights/
cp -r ./checkpoints/ FaceSwapping/supervision/reenactment/LIA/
cd ..
```

#### 2.2 DeepFill_V2 (from [DeepFill_V2 GitHub](https://github.com/JiahuiYu/generative_inpainting/tree/v2.0.0))

```shell
cd ./deepfillv2_weights/
cp -r ./weights/ FaceSwapping/supervision/restoration/DeepFill_V2/
cd ..
```

#### 2.3 GPEN (from [GPEN GitHub](https://github.com/yangxy/GPEN))

```shell
cd ./gpen_weights/
cp -r ./weights/ FaceSwapping/supervision/restoration/GPEN/
cd ..
```

#### 2.4 DML_CSR (from [DML_CSR GitHub](https://github.com/deepinsight/insightface/tree/master/parsing/dml_csr))

```shell
cd ./dmlcsr_weights/
cp -r ./weights/ FaceSwapping/supervision/parsing/dml_csr/
cd ..
```

#### 2.5 faceiqa (from [SER_FIQ GitHub](https://github.com/pterhoer/FaceImageQuality))

```shell
cd ./faceiqa_weights/
cp -r ./model/ FaceSwapping/modules/third_party/faceiqa/insightface/
cd ..
```

### Step 3. Running

```shell
python3 triplet_synthesizer.py \
    --save_dataset Your/Folder/to/Save/Synthetic/Results/
```