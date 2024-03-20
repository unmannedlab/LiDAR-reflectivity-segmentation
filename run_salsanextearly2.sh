#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d /home/ubuntu/calintensity/Datasets/rellis_lidar_nr/  -ac ./config/arch/salsanextearly_rxyzi.yml -dc ./config/labels/rellis.yaml -n salsanextearly_rxyzi_attach -l ./logs -p ""
