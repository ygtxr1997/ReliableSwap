## 3. Testing


### 3.1. Image- and Video-based Inference

```shell
cd ReliableSwap/inference
chmod +x faceshifter_infer.sh
bash ./faceshifter_infer.sh
# uncomment the codes in 'inference/faceshifter_infer.sh' to run other demos
# demo0: faceshifter vanilla, image infer
# demo1: faceshifter vanilla, video infer
# demo2: InfoSwap, image infer
# or directly run 'python3 faceshifter_infer.py' by giving other args
```

Please refer to our [Hugging Face Demo](https://huggingface.co/spaces/ygtxr1997/ReliableSwap_Demo) for 
more inference examples.

### 3.2. Using GPEN Image Enhancement (Optional)

```shell
cd ReliableSwap/supervision/restoration/GPEN
python3 infer_video.py --indir your/path/to/input_video.mp4 --outdir output/folder/
```