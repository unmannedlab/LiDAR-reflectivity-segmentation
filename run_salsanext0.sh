#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic;  ./train.py -d /media/usl/Data/Dataset/Semantic-poss/semantic_poss_reflectivity/sequences/  -ac ./config/arch/poss/salsanext_rxyzi.yml -dc ./config/labels/poss.yaml -n poss_rxyzi -l ./logs -p ""
