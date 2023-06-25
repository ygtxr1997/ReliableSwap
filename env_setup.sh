#!/bin/bash

echo "Installing python environment"
pip install -r requirements.txt

echo "Installing NMS for PIPNet"
cd inference/PIPNet/FaceBoxesV2/utils || exit
chmod +x make.sh
bash ./make.sh
cd - || exit

echo "[Note] You may go to './supervision' and run 'bash ./env_setup.sh' for generating triplets by yourself,
more details can be found in './supervision/readme.md'."
