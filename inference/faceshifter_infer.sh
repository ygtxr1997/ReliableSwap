#!/bin/bash

  echo "demo0: faceshifter vanilla, image infer"
  python3 faceshifter_infer.py \
          -s examples/20015.jpg \
          -t examples/08095.jpg \
          -o examples/ \
          -cp ../trainer/faceshifter/extracted_ckpt/faceshifter_vanilla_3.pth \
          --align_source ffhq \
          --align_target ffhq \

#  echo "demo1: faceshifter vanilla, video infer"
#  python3 faceshifter_infer.py \
#          -s examples/source_red.jpg \
#          -t examples/target.mp4 \
#          -o examples/ \
#          -cp ../trainer/faceshifter/extracted_ckpt/faceshifter_vanilla_3.pth \
#          --align_source ffhq \
#          --align_target ffhq \

#  echo "demo2: InfoSwap, image infer"
#  python3 faceshifter_infer.py \
#          -s examples/20015.jpg \
#          -t examples/08095.jpg \
#          -o examples/ \
#          -cp infoswap \
#          --align_source ffhq \
#          --align_target ffhq \
