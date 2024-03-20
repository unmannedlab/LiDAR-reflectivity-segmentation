#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d /home/ubuntu/calintensity/Datasets/data_reflectivity_velodyne_v2/dataset/sequences/  -ac ./config/arch/kitti/early_l1_detach.yml -dc ./config/labels/semantic-kitti.yaml -n early_l1_detachkitti -l ./logs -p ""
