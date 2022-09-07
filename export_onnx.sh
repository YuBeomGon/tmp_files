#!/bin/bash
# import argparse
# import os

# parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
# parser.add_argument('--input_checkpoint', default='./checkpoints/swin_mlp_tiny_baseline_att_384x384_A.json.checkpoints/epoch_240.pth'
#                     , type=str, help='directory for model checkpoint')
# parser.add_argument('--input_topology', default='utils/human_pose_2.json'
#                     , type=str, help='directory for input topology')
# parser.add_argument('--input_width', default=384, type=int, metavar='N', help='width')
# parser.add_argument('--input_height', default=384, type=int, metavar='N', help='width')
# parser.add_argument('--output_model', default='./exported/swin_mlp_tiny_384x384_epoch_240.onnx'
#                     , type=str, help='directory for model checkpoint')

input_width=384
input_height=384
#input_model='swin_mlp_tiny'
input_model='densenet169'
num_epoch=240
deep_stream_pose_apps_dir='/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream_pose_estimation/'

input_checkpoint='./checkpoints/'$input_model'_baseline_att_'$input_width'x'$input_height'_A.json.checkpoints/epoch_'$num_epoch'.pth'
input_topology='utils/human_pose_2.json'
output_model='./exported/'$input_model'_'$input_width'x'$input_height'_epoch_'$num_epoch'.onnx'

echo $input_checkpoint
echo $output_model

python utils/export_for_isaac.py --input_checkpoint $input_checkpoint \
                                 --input_model $input_model'_baseline_att' --input_topology $input_topology \
                                 --input_width $input_width --input_height $input_height \
                                 --output_model $output_model
                                 
sudo cp $output_model $deep_stream_pose_apps_dir'onnx/'                         
