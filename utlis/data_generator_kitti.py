import open3d as o3d
import numpy as np

#import numpy as np
#from plyreader import PlyReader
#import matplotlib.pyplot as plt
import os
from multiprocessing.dummy import Pool
import sys
#sys.path.append('/media/moonlab/sd_card/')
import tqdm

import alpha_predictor.alpha_model as alpha_model
import torch
import torch.backends.cudnn as cudnn
from time import time
import copy

device_o3d = o3d.core.Device("cuda:0")
dtype = o3d.core.float32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)

#root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
models = alpha_model.alpha()
models.to(device)
#print(models)
models.load_state_dict(torch.load('./alpha_predictor/models/best_model_mega_tanh.pth')['model_state_dict'],strict = True)

fit = np.load('./fit_usl_sidewalk.npy',allow_pickle=True)
p = np.poly1d(fit)

fit_eta = np.load('./eta_fit_grass.npy',allow_pickle = True)
p_eta = np.poly1d(fit_eta)

def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    #print(obj.shape)
    return obj

def alpha_predictor(input):
    data = torch.Tensor(input)
    models.eval()
    inp = data.to(device)
    out = models(inp)
    angles = out.detach().cpu().numpy()
    angles = np.array(angles).flatten()
    return angles

def convert_ply2bin(bin_path):
    #pr = PlyReader()
    #plydata = pr.open(ply_path)
    #vertex =plydata['vertex']
    #print('lol')
    
    raw_points = load_bin('/media/usl/Data/Dataset/semantic-kitti/data_odometry_velodyne/dataset/sequences/'+k+'/velodyne/'+bin_path)
    labels = np.fromfile('/media/usl/Data/Dataset/semantic-kitti/data_odometry_labels/dataset/sequences/'+k+'/labels/'+bin_path[:-3]+'label',dtype=np.uint32)
    
    x,y,z,ins= raw_points[:,0],raw_points[:,1],raw_points[:,2],(raw_points[:,3]*255).astype(np.uint8)
    dis = np.square(x)+np.square(y)+np.square(z)
    dist = np.abs(np.sqrt(dis))

    ind = np.where(dist > 50)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    #rang = np.delete(rang,list(ind))
    dist = np.delete(dist,list(ind))
    dis = np.delete(dis,list(ind))
    labels = np.delete(labels,list(ind))

    # intensity_os = ins.copy()
    # intensity_os = intensity_os*p(dist)
    # intensity_os = abs(intensity_os)

    pc = o3d.t.geometry.PointCloud(device_o3d)

    points = np.zeros((len(x),3))
    points[:,0], points[:,1], points[:,2] = x, y, z
    pc.point.positions = o3d.core.Tensor(points,dtype,device_o3d)
    pc.estimate_normals(radius = 0.3, max_nn = 10)

    nm = np.asarray(pc.point.normals.cpu().numpy())
    #print(nm)
    nm[:,2] = np.abs(nm[:,2]) ## converting the rogue normals eg: downward normal vectors for grass
    #print('lol')
    angles = alpha_predictor(nm)

    cal_intensity = ins/np.cos(angles)
    reflectivity = copy.deepcopy(cal_intensity)
    reflectivity = reflectivity
    if np.max(reflectivity)>=1200:
        print('lol')

    #print(np.max(reflectivity))
    # index = np.where(dist<12)[0]
    # cal_intensity[index] = cal_intensity[index]/p_eta(dist[index])*1.6
    # cal_intensity = cal_intensity/np.max(cal_intensity)

    points_n = np.zeros((len(x),6),dtype = np.float32)
    points_n[:,0], points_n[:,1], points_n[:,2], points_n[:,3],points_n[:,4], points_n[:,5] = x,y,z,ins/255, reflectivity/1200, reflectivity/1200
    points_n.tofile('/media/usl/Data/Dataset/semantic-kitti/data_reflectivity_velodyne_v3/dataset/sequences/'+k+'/velodyne/'+bin_path)
    labels.tofile('/media/usl/Data/Dataset/semantic-kitti/data_reflectivity_velodyne_v3/dataset/sequences/'+k+'/labels/'+bin_path[:-3]+'label')
    
    #print(bin_path)
    #return x, y, z, cal_intensity


if __name__ =="__main__":
    dirs = ['00', '01','02','03','04','05','06','07','08','09','10']
    for k in dirs:
        files = os.listdir('/media/usl/Data/Dataset/semantic-kitti/data_odometry_velodyne/dataset/sequences/'+k+'/velodyne/')
        #convert_ply2bin(files[5])
        with Pool() as pool:
            r = list(tqdm.tqdm(pool.imap_unordered(convert_ply2bin,files),total = len(files)))
    #x, y, z, ins, dis, angles = convert_ply2bin('/home/usl/Desktop/random/000000.bin')
    # ind = np.where(cal_intensity > 2000)[0]
    # ins = np.delete(ins,list(ind))
    # x = np.delete(x,list(ind))
    # y = np.delete(y,list(ind))
    # z = np.delete(z,list(ind))
    # cal_intensity = (np.delete(cal_intensity,list(ind))/2000*255)
    
    #np.save('./000000_cust1.npy', points,allow_pickle=True)
    #cal_intensity = (cal_intensity/max(cal_intensity)*255).astype(np.int8)
    #print(cal_intensity)
