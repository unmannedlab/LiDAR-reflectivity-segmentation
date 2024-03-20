#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d /home/ubuntu/calintensity/Datasets/rellis_lidar_nr/  -ac ./config/arch/early_l1_atch.yml -dc ./config/labels/rellis.yaml -n early_l1_atch -l ./logs -p ""
