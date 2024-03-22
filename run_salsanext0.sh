#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d path/to/dataset/rellid_lidar_nr/  -ac ./config/arch/rellis/salsanext_rxyzi.yml -dc ./config/labels/rellis.yaml -n rellis_rxyzi -l ./logs -p ""
