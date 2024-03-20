#import open3d
import numpy as np

import numpy as np
#from plyreader import PlyReader
import matplotlib.pyplot as plt
import os

import sys
#sys.path.append('/media/moonlab/sd_card/')

import alpha_predictor.alpha_model as alpha_model
import torch
import torch.backends.cudnn as cudnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmak = False
cudnn.deterministic = True
cudnn.enabled = True
torch.cuda.set_device(0)

#root = '/media/moonlab/sd_card/Rellis_3D_lidar_example/'
models = alpha_model.alpha()
models.to(device)
#print(models)
models.load_state_dict(torch.load('./alpha_predictor/models/best_model_mega_tanh.pth')['model_state_dict'])

def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
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
    points = load_bin(bin_path)
    #pc = open3d.geometry.PointCloud()
    x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,3]*65535).astype(np.int16)
    rang = np.sqrt(x**2+y**2+z**2)
    dis = x**2 + y**2 + z**2

    ind = np.where(rang < 1)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    rang = np.delete(rang,list(ind))
    dis = np.delete(dis,list(ind))


    points_n = np.zeros((len(x),3))
    points_n[:,0],points_n[:,1],points_n[:,2] = x , y, z
    ld_vec = points_n/np.sqrt(dis[:,np.newaxis])
    #pc.points = open3d.utility.Vector3dVector(points)
    #pc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = 0.5,max_nn = 10))
    #open3d.visualization.draw_geometries([pc])
    #nm = np.asarray(pc.normals)
    angles = alpha_predictor(ld_vec)
    return x, y, z, ins, dis, angles

if __name__ =="__main__":
    x, y, z, ins, dis, angles = convert_ply2bin('/home/usl/Desktop/random/000000.bin')
    cal_intensity = ins*dis/np.cos(angles)/100
    ind = np.where(cal_intensity > 2000)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    cal_intensity = (np.delete(cal_intensity,list(ind))/2000*255)
    points = np.zeros((len(x),4),dtype = np.float32)
    points[:,0], points[:,1], points[:,2], points[:,3] = x,y,z,cal_intensity
    np.save('./000000_cust1.npy', points,allow_pickle=True)
    #cal_intensity = (cal_intensity/max(cal_intensity)*255).astype(np.int8)
    #print(cal_intensity)