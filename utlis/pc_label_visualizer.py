import numpy as np
import open3d
color_palette = {
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    12: {"color": [255, 0, 0],  "name": "building"},
    15: {"color": [102, 0, 0],  "name": "log"},
    17: {"color": [204, 153, 255],  "name": "person"},
    18: {"color": [102, 0, 204],  "name": "fence"},
    19: {"color": [255, 153, 204],  "name": "bush"},
    23: {"color": [197,197,197],  "name": "concrete"},
    27: {"color": [41, 121, 255],  "name": "barrier"},
    31: {"color": [134, 255, 239],  "name": "puddle"},
    33: {"color": [99, 66, 34],  "name": "mud"},
    34: {"color": [110, 22, 138],  "name": "rubble"}
}



def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    return obj

points = load_bin('/home/usl/Desktop/rellis_lidar_nr/00000/os1_cloud_node_kitti_bin/000104.bin')
labels = np.fromfile('/home/usl/Desktop/random/000104_xyzn.label',dtype = np.int32,count = -1)
temp = labels.copy()
image = np.zeros((len(labels),3))
for k, v in  color_palette.items():
        #print(v['color'])
        image[temp == k, :] = v["color"]
pc = open3d.geometry.PointCloud()
pc.points = open3d.utility.Vector3dVector(points[:,:3])
pc.colors = open3d.utility.Vector3dVector(image/255.)
open3d.visualization.draw_geometries([pc])