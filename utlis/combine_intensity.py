import numpy as np
import os
from tqdm import tqdm

if __name__=="__main__":
    rellis_root = "path/to/dataset/rellis_lidar_nr"
    seqs = np.arange(5)
    intensity_type = {"intensity": "os1_cloud_node_kitti_bin",
                      "reflectivity":"os1_cloud_node_kitti_reflec",
                      "near_range":"os1_cloud_node_kitti_nr"}
    for seq in tqdm(seqs):
        seq_dir = os.path.join(rellis_root, f"{seq:05d}")
        intensity_dir = os.path.join(seq_dir, intensity_type["intensity"])
        reflectivity_dir = os.path.join(seq_dir, intensity_type["reflectivity"])
        near_range_dir = os.path.join(seq_dir, intensity_type["near_range"])
        combined_intensity_dir = os.path.join(seq_dir, "combined_intensity")
        if not os.path.exists(combined_intensity_dir):
            os.makedirs(combined_intensity_dir)
        file_list = os.listdir(intensity_dir)
        file_list.sort()
        for file in tqdm(file_list):
            intensity = np.fromfile(os.path.join(intensity_dir, file), dtype=np.float32).reshape(-1, 4)
            dists = np.linalg.norm(intensity[:,:3], axis=1)
            intensity = intensity[dists>=1]
            reflectivity = np.fromfile(os.path.join(reflectivity_dir, file), dtype=np.float32).reshape(-1, 4)
            near_range = np.fromfile(os.path.join(near_range_dir, file), dtype=np.float32).reshape(-1, 4)
            combined_intensity = np.concatenate([intensity, reflectivity[:,-1].reshape(-1,1), near_range[:,-1].reshape(-1,1)], axis=1)
            combined_intensity = combined_intensity.astype(np.float32)
            combined_intensity.tofile(os.path.join(seq_dir, "combined_intensity", file))
    
