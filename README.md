## Implementation of "Reflectivity is All You Need!: Advancing LiDAR Semantic Segmentation
#### The repository is under frequent updation.  

### Summary

##### This repository explores the benefits of incorporating calibrated intensity (reflectivity) in learning-based LiDAR semantic segmentation frameworks. By leveraging reflectivity alongside raw intensity measurements, our model exhibits improved performance, particularly in off-road scenarios.

![Results Illustration](./images/result.png)

### Generating reflectivity data
```
python utils/data_generator.py
```
##### Modify the dataset and output path.

#### Modified Rellis-3D dataset used for training and testing. [Download.](https://drive.google.com/file/d/1nWOecnBa6WugoBl-JnFZzV2s9ogXnZw_/view?usp=sharing) 

![Dataset Illustration](./images/irs.png)


