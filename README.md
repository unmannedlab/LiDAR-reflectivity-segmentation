## Implementation of "Reflectivity is All You Need!: Advancing LiDAR Semantic Segmentation
#### The repository is under frequent updation.  

### Summary

##### This repository explores the benefits of incorporating calibrated intensity (reflectivity) in learning-based LiDAR semantic segmentation frameworks. By leveraging reflectivity alongside raw intensity measurements, our model exhibits improved performance, particularly in off-road scenarios.

![Results Illustration](./images/result.png)
##### *rxyzi* represents model trained on raw intensity data.
##### *rxyzn* represents model trained on reflectivity data.

### Generating reflectivity data
```
python utils/data_generator.py
```
Modify the dataset and output file path.

#### Modified Rellis-3D dataset used for training and testing. [Download.](https://drive.google.com/file/d/1nWOecnBa6WugoBl-JnFZzV2s9ogXnZw_/view?usp=sharing) 

![Dataset Illustration](./images/irs.png)
(a) Illustrates spherical projection of LiDAR with raw intensity as pixel values. (b) Calibrated for *range* and *angle of incidence*. (c) Calibrated for *range*, *angle of incidence* and *near-range effect*. 

### SalsaNext
Original [Salsanext](https://github.com/TiagoCortinhal/SalsaNext) and modified versions config files can be found in:
```
cd ./train/tasks/semantic/config/arch
```
The modified versions of salsanext mentioned in paper is:
```
salsanext_rxyzi.yml
salsanext_rxyzirn.yml
salsanext_rxyzn.yml
early_ga_detach.yml (*learning reflectivity*)
```
![SalsaNext_model](./images/network.pdf)

### Evaluate SalsaNext

