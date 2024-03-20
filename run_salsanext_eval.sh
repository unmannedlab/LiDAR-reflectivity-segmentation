#!/bin/sh
export CUDA_VISIBLE_DEVICES="0"
cd ./train/tasks/semantic
python infer2.py -d /home/usl/Desktop/rellis_lidar_nr/ -l /home/usl/Desktop/physics_salsanext/train/tasks/semantic/logs/ -s test -m /home/usl/Desktop/physics_salsanext/train/tasks/semantic/logs/logs/2024-2-27-12-04salsanext_rxyzn/ -u False
python evaluate_iou.py -d /home/usl/Desktop/rellis_lidar_nr/ -p /home/usl/Desktop/physics_salsanext/train/tasks/semantic/logs/salsa/ -ac /home/usl/Desktop/physics_salsanext/train/tasks/semantic/logs/logs/2024-2-27-12-04salsanext_rxyzn/arch_cfg.yaml