#!/bin/sh
export CUDA_VISIBLE_DEVICES="1"
cd ./train/tasks/semantic;  ./train.py -d /media/peng/7c3e5329-39d0-4b7d-a8fe-af5ad9c0d370/Datasets/data_reflectivity_velodyne_v2/dataset/sequences/  -ac ./config/arch/kitti/salsanext_rxyzi.yml -dc ./config/labels/semantic-kitti.yaml -n kitti_rxyzi -l ./logs -p ""
