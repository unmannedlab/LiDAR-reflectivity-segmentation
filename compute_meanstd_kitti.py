import numpy as np
import os
from tqdm import tqdm

if __name__=="__main__":
    kiit_root = "/media/peng/7c3e5329-39d0-4b7d-a8fe-af5ad9c0d370/Datasets/data_reflectivity_velodyne_v2/dataset/sequences"
    all_combined_intensity = []
    seqs= np.arange(0,11)
    for seq in tqdm(seqs):
        seq_dir = os.path.join(kiit_root, f"{seq:02d}")
        combined_intensity_dir = os.path.join(seq_dir, "velodyne")
        file_list = os.listdir(combined_intensity_dir)
        file_list.sort()
        for file in tqdm(file_list):
            combined_intensity = np.fromfile(os.path.join(combined_intensity_dir, file), dtype=np.float32).reshape(-1, 6)
            #dist = np.linalg.norm(combined_intensity[:,:3], axis=1)
            #combined_intensity[dist>50, 4] =combined_intensity[dist>50, 3]
            #combined_intensity[dist>50, 5] =combined_intensity[dist>50, 3]

            #combined_intensity[:,3:] = combined_intensity[:,3:]/np.max(combined_intensity[:,3:], axis=0)
            # combined_intensity.tofile(os.path.join(combined_intensity_dir, file))
            all_combined_intensity.append(combined_intensity[:,:3])
    all_combined_intensity = np.concatenate(all_combined_intensity, axis=0)
    range = np.linalg.norm(all_combined_intensity[:,:3], axis=1)
    x = all_combined_intensity[:,0]
    y = all_combined_intensity[:,1]
    z = all_combined_intensity[:,2]
    intensity = np.zeros(10)

    reflectivity = np.zeros(10)
    #print(np.max(reflectivity), np.min(reflectivity))

    near_range = np.zeros(10)

    
    with open("img_meanstd_kitti.txt", "w") as f:
        f.write("img_means:\n")
        f.write("#range,x,y,z,intensity,reflectivity,near_range\n")
        f.write(f" -{range.mean()}\n")
        f.write(f" -{x.mean()}\n")
        f.write(f" -{y.mean()}\n")
        f.write(f" -{z.mean()}\n")
        f.write(f" -{intensity.mean()}\n")
        f.write(f" -{reflectivity.mean()}\n")
        f.write(f" -{near_range.mean()}\n")
        f.write("img_stds:\n")
        f.write("#range,x,y,z,intensity,reflectivity,near_range\n")
        f.write(f" -{range.std()}\n")
        f.write(f" -{x.std()}\n")
        f.write(f" -{y.std()}\n")
        f.write(f" -{z.std()}\n")
        f.write(f" -{intensity.std()}\n")
        f.write(f" -{reflectivity.std()}\n")
        f.write(f" -{near_range.std()}\n")
    print("mean and std saved to img_meanstd.txt")