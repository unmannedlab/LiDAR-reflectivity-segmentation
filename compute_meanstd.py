import numpy as np
import os
from tqdm import tqdm

if __name__=="__main__":
    rellis_root = "/media/usl/Data/Dataset/Semantic-poss/semantic_poss_reflectivity/sequences"
    seqs = np.arange(5)
    intensity_type = {"intensity": "os1_cloud_node_kitti_bin",
                      "reflectivity":"os1_cloud_node_kitti_reflec",
                      "near_range":"os1_cloud_node_kitti_nr"}
    all_combined_intensity = []
    for seq in seqs:
        seq_dir = os.path.join(rellis_root, f"{seq:02d}")
        combined_intensity_dir = os.path.join(seq_dir, "velodyne")
        file_list = os.listdir(combined_intensity_dir)
        file_list.sort()
        for file in file_list:
            combined_intensity = np.fromfile(os.path.join(combined_intensity_dir, file), dtype=np.float32).reshape(-1, 6)
            all_combined_intensity.append(combined_intensity)
    all_combined_intensity = np.concatenate(all_combined_intensity, axis=0)
    range = np.linalg.norm(all_combined_intensity[:,:3], axis=1)
    x = all_combined_intensity[:,0]
    y = all_combined_intensity[:,1]
    z = all_combined_intensity[:,2]
    intensity = all_combined_intensity[:,3]

    reflectivity = all_combined_intensity[:,4]

    near_range = all_combined_intensity[:,5]

    # save mean and std to file 
    # img_means: 
    # #range,x,y,z,intensity,reflectivity,near_range
    #   - 4.84649722
    #   - -0.187910314
    #   - 0.193718327
    #   - -0.246564824
    #   - 0.010744918
    # img_stds:     
    # #range,x,y,z,intensity,reflectivity,near_range
    #   - 6.05381850
    #   - 5.61048984
    #   - 5.27298844
    #   - 0.849105890
    #   - 0.0069436138
    
    with open("img_meanstd.txt", "w") as f:
        f.write("img_means:\n")
        f.write("#range,x,y,z,intensity,reflectivity,near_range\n")
        f.write(f"      - {range.mean()}\n")
        f.write(f"      - {x.mean()}\n")
        f.write(f"      - {y.mean()}\n")
        f.write(f"      - {z.mean()}\n")
        f.write(f"      - {intensity.mean()}\n")
        f.write(f"      - {reflectivity.mean()}\n")
        f.write(f"      - {near_range.mean()}\n")
        f.write("img_stds:\n")
        f.write("#range,x,y,z,intensity,reflectivity,near_range\n")
        f.write(f"      - {range.std()}\n")
        f.write(f"      - {x.std()}\n")
        f.write(f"      - {y.std()}\n")
        f.write(f"      - {z.std()}\n")
        f.write(f"      - {intensity.std()}\n")
        f.write(f"      - {reflectivity.std()}\n")
        f.write(f"      - {near_range.std()}\n")
    print("mean and std saved to img_meanstd.txt")