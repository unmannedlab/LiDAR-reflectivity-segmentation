cd ./train/tasks/semantic
python infer2.py -d /path/to/dataset/ -l /path/to/logs -m /path/to/pretrained/model -n rxyzn 
python evaluate_iou.py -d /path/to/dataset/ -p /path/to/infered/data
