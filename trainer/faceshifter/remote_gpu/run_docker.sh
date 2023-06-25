#!/bin/bash
image_full_name=$1
if [ -z ${image_full_name} ]
then
   echo "please input a image name. eg: mirrors.tencent.com/star_library/g-tlinux2.2-python3.6-cuda9.0-cudnn7.6-tf1.12:latest    \n"
   exit 0
fi

cmd=$2
if [ -z "${cmd}" ]
then
   echo "please input your train command, eg: python3.6 /apdcephfs/private_YOURRTX/train/train.py  --dataset_dir=/apdcephfs/private_YOURRTX/data     \n"
   exit 0
fi

echo ${image_full_name}
echo ${cmd}
docker run -it --gpus all --network=host --ipc=host -v /apdcephfs/:/apdcephfs/ -v /apdcephfs_cq2/:/apdcephfs_cq2/ ${image_full_name}  ${cmd}

#如果docker -v版本小于19.03，请使用以下docker run命令
#docker run -it -e NVIDIA_VISIBLE_DEVICES=all --network=host -v /apdcephfs/:/apdcephfs/  ${image_full_name}  ${cmd}

# mirrors.tencent.com/ss_face/faceswap:latest
# mirrors.tencent.com/gavinyuan/faceswap:1.?

# Save image
# docker commit {container_id} {mirrors.tencent.com/YOUR_NAMESPACE/YOUR_IMAGENAME}:{tag}

# Push image
# docker push {mirrors.tencent.com/YOUR_NAMESPACE/YOUR_IMAGENAME}:{tag}

# Debug command
# ./run_docker.sh mirrors.tencent.com/gavinyuan/faceswap:1.? /bin/bash -c "python3 /apdcephfs_cq2/share_1290939/gavinyuan/code/FaceShifter/faceswap/trainer/faceshifter/faceshifter_trainer.py -n hello -bs 1"

# jizhi attach
# jizhi_client exec TEG_AILab_CVC_chongqing525342E2146443B3A 899688f3801ca2790180bd3597bd17f8 bash