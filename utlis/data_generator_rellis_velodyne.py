import open3d
import numpy as np

#import numpy as np
#from plyreader import PlyReader
#import matplotlib.pyplot as plt
import os
from multiprocessing.dummy import Pool
import sys
#sys.path.append('/media/moonlab/sd_card/')

import alpha_predictor.alpha_model as alpha_model
import torch
import torch.backends.cudnn as cudnn
import spherical_projection
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)
import copy

#root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
models = alpha_model.alpha()
models.to(device)
#print(models)
models.load_state_dict(torch.load('./alpha_predictor/models/best_model_mega_tanh.pth')['model_state_dict'],strict = True)
vel2os_fit = np.load('./fit_tree_vel.npy',allow_pickle = True)
vel2os = np.poly1d(vel2os_fit)

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

def vel2ouster(intensity, distance,vel2os):
    qot = vel2os(distance)
    intense = intensity*qot
    # f_qt = []
    # for k in prange(len(intensity)):
    #     intes = intensity[k]
    #     dic = distance[k]
    #     qot = vel2os(dic)
    #     intes = intes*qot
    #     f_qt.append(intes)
    return intense

def convert_ply2bin(bin_path):
    #pr = PlyReader()
    #plydata = pr.open(ply_path)
    #vertex =plydata['vertex']
    #print('lol')
    points = load_bin('/media/usl/Data/Dataset/Rellis-3D/Rellis_3D_vel_cloud_node_kitti_bin/Rellis-3D/'+dir_index+'/vel_cloud_node_kitti_bin/'+bin_path)
    labels = np.fromfile('/media/usl/Data/Dataset/Rellis-3D/Rellis_3D_vel_cloud_node_semantickitti_label_id/Rellis-3D/'+dir_index+'/vel_cloud_node_semantickitti_label_id/'+bin_path[:-3]+'label',dtype = np.int32,count = -1)
    pc = open3d.geometry.PointCloud()
    x,y,z,ins= points[:,0],points[:,1],points[:,2],points[:,3].astype(np.uint8)
    #print(ins.max())
    intensity = copy.deepcopy(ins)
    rang = np.sqrt(x**2+y**2+z**2)
    dis = x**2 + y**2 + z**2

    ins = vel2ouster(ins,rang,vel2os)
    reflectivity = copy.deepcopy(ins)
    #print(reflectivity.max())

    ind = np.where(rang < 0.5)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    rang = np.delete(rang,list(ind))
    dis = np.delete(dis,list(ind))
    labels = np.delete(labels,list(ind))
    intensity = np.delete(intensity,list(ind))
    reflectivity = np.delete(reflectivity,list(ind))

    ind = np.where(rang >50)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    rang = np.delete(rang,list(ind))
    dis = np.delete(dis,list(ind))
    labels = np.delete(labels,list(ind))
    intensity = np.delete(intensity,list(ind))
    reflectivity = np.delete(reflectivity,list(ind))


    index = np.where(rang<12)[0]
    ins[index] = ins[index]/p(rang[index])*1.8


    points_n = np.zeros((len(x),3))
    points_n[:,0],points_n[:,1],points_n[:,2] = x , y, z
    #ld_vec = points_n/rang[:,np.newaxis]
    pc.points = open3d.utility.Vector3dVector(points_n)
    pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.3,max_nn = 10))
    #open3d.visualization.draw_geometries([pc])
    nm = np.asarray(pc.normals)
    nm[:,2] = np.abs(nm[:,2])
    #print(len(nm))
    angles = alpha_predictor(nm)
    cal_intensity = ins*dis/np.cos(angles)
    reflectivity = reflectivity*dis/np.cos(angles)
    cal_intensity = cal_intensity/cal_intensity.max()
    reflectivity = reflectivity/reflectivity.max()
    points_f = np.zeros((len(x),6),dtype = np.float32)
    points_f[:,0], points_f[:,1], points_f[:,2], points_f[:,3], points_f[:,4], points_f[:,5] = x,y,z, intensity/255, reflectivity, cal_intensity
    #spherical_projection.main(points_n)
    points_f.tofile('/media/usl/Data/Dataset/Rellis-3D/Rellis_3d_velodyne_reflectivity/'+dir_index+'/os1_cloud_node_kitti_bin/'+bin_path)
    labels.tofile('/media/usl/Data/Dataset/Rellis-3D/Rellis_3d_velodyne_reflectivity/'+dir_index+'/os1_cloud_node_semantickitti_label_id/'+bin_path[:-3]+'label')
    #print(bin_path)
    #return x, y, z, cal_intensity


if __name__ =="__main__":
    files_dir= ['00000','00001','00002','00003','00004']
    fit = np.load('./eta_fit_grass.npy',allow_pickle = True)
    p = np.poly1d(fit)
    for dir_index in files_dir:
        print('Started Sequence:',dir_index)
        files = os.listdir('/media/usl/Data/Dataset/Rellis-3D/Rellis_3D_vel_cloud_node_kitti_bin/Rellis-3D/'+dir_index+'/vel_cloud_node_kitti_bin')
    #convert_ply2bin(files[0])
        with Pool() as pool:
            r = list(tqdm.tqdm(pool.imap_unordered(convert_ply2bin,files),total=len(files)))
        print('Completede Sequence: ',dir_index)
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
