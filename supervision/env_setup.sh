#!/bin/bash

GPU_ARCH=$1

cd parsing/inplace_abn || exit
export TORCH_CUDA_ARCH_LIST=${GPU_ARCH}
export IABN_FORCE_CUDA=1
pip uninstall -y inplace-abn
python3 setup.py install
cd ../..

cd restoration/neuralgym || exit
pip uninstall -y neuralgym
python3 setup.py install
cd ../..

printf "[Modules successfully installed.]\n"
pip list |grep inplace-abn
pip list |grep neuralgym

printf "[Checking GPU environment (arch=%s)...]\n" "${GPU_ARCH}"
python3 ../tools/env_check.py

# On Turing: bash ./env_setup.sh 7.5
# On Ampere: bash ./env_setup.sh 8.0