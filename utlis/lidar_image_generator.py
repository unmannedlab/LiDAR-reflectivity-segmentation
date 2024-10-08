from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    return obj

def image_projection(img,u,v,ins):
    for i in range(len(u)):
        img[u[i]][int(v[i])] = ins[i]
    return img

def processing(points, ins = None, ref = None, rang = None):
    x,y,z= points[:,0],points[:,1],points[:,2]
    fov_up = 0.392
    fov_down = -0.392
    rang = np.sqrt(x**2+y**2+z**2)
    row_scale = 64
    col_scale = 2047
    img = np.zeros((64,2048), dtype = np.int8)
    if ins is not None:
        label = (ins/np.max(ins)*255).astype(np.int8)
    elif ref is not None:
        label = (ref/np.mean(ref)*80).astype(np.int8)
    else :
        label = (rang/np.max(rang)*255).astype(np.int8)

    u = (row_scale*(-((np.arcsin(z/rang)+fov_down)/0.784))).astype(np.uint16)
    v = col_scale*(0.5*((np.arctan2(y,x)/3.141)+1))

    image = image_projection(img,u,v,label)
    return image

if __name__ == "__main__":
    # Load the image
    dir_fol = ['00000','00001','00002','00003','00004']
    for j in dir_fol:
        files = os.listdir('/path/to/rellis_lidar_nr/'+ j +'/os1_cloud_node_kitti_bin')
        for i in tqdm(files):
            points = load_bin('/path/to/rellis_lidar_nr/'+ j +'/os1_cloud_node_kitti_bin/'+ i)
            x,y,z,ins,_,nr = points[:,0],points[:,1],points[:,2],points[:,3],points[:,4],points[:,5]

            ranged = np.sqrt(x**2+y**2+z**2)
            ind = np.where(ranged == 0)[0]
            ins = np.delete(ins,list(ind))
            x = np.delete(x,list(ind))
            y = np.delete(y,list(ind))
            z = np.delete(z,list(ind))
            ranged = np.delete(ranged,list(ind))
            nr = np.delete(nr,list(ind))

            intensity_image = processing(points, ins = ins)
            reflectivity_image = processing(points, ref = nr)
            range_image = processing(points, rang = ranged)

            image = np.zeros((64,2048,3), dtype = np.uint8)
            image[:,:,0] = intensity_image
            image[:,:,1] = reflectivity_image
            image[:,:,2] = range_image

            img = Image.fromarray(image)
            img.save('/save/to/data/'+ j +'/image/'+i[:-4]+'.png')
            
