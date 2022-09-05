#!/bin/bash

arch='resnet18'
num_workers=`grep -c processor /proc/cpuinfo`
deephome='/opt/nvidia/deepstream/deepstream-6.1/sources/apps/sample_apps/deepstream_pose_estimation'

# https://docs.docker.com/engine/reference/run/#ipc-settings---ipc
# set --ipc=host for solving out of shared memory 

# docker run --gpus all -it --ipc=host --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_contra --arch $arch --workers $num_workers

docker run --gpus all -it --ipc=host --rm -v /home/beomgon/Downloads:$deephome/Downloads beomgon/deepstream:inference bash
# docker run --gpus all -it --ipc=host --rm  -v /home/beomgon/Downloads:$deephome/Downloads beomgon/deepstream:inference bash
#  docker run --gpus all -it --ipc=host --rm -v ~/Downloads:/opt/nvidia/deepstream/deepstream-6.1/Downloads -v $PWD:$deephome beomgon/deepstream:inference bash

#docker run --gpus all -it --pid=host --rm -v $PWD:/project/pose -v ~/dataset:/project/pose/dataset -p 8888:8888 beomgon/deepstream:pose_base.notebook
# nvcr.io/nvidia/tensorrt:xx.xx-py3
