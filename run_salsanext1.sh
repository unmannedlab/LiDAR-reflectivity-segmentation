#!/bin/sh
export CUDA_VISIBLE_DEVICES="1"
cd ./train/tasks/semantic;  ./train.py -d /path/to/dataset/rellis_lidar_nr  -ac ./config/arch/rellis/salsanext_rxyzn.yml -dc ./config/labels/rellis.yaml -n rellis_rxyzn -l ./logs -p ""
