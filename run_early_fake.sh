#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d /home/ubuntu/calintensity/Datasets/rellis_lidar_nr/  -ac ./config/arch/early_fake.yml -dc ./config/labels/rellis.yaml -n early_fake -l ./logs -p ""
