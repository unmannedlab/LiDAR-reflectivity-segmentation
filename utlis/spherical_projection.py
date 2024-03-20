import cv2
import open3d
import numpy as np
import math

def load_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    return obj

def image_projection(img,u,v,ins):
    for i in range(len(u)):
        img[u[i]][int(v[i])] = ins[i]
    return img
    #cv2.imwrite('./reflectivity_proj.png',img)
    #cv2.imshow('lol',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def load_npy(npy_path):
    obj = np.load(npy_path,allow_pickle=True)
    return obj

def main(points=None):
    zoints = load_bin('/home/usl/Desktop/random/000000_xyzirn.bin')
    #points = load_npy('/home/usl/Desktop/random/000000_cust1.npy')
    print(points.shape)
    #x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,3]*65535).astype(np.int16)
    x,y,z,ins, ref, nr = points[:,0],points[:,1],points[:,2],points[:,3], points[:,4], points[:,5]
    fov_up = 0.392
    fov_down = -0.392
    rang = np.sqrt(x**2+y**2+z**2)
    row_scale = 64
    col_scale = 2047
    ins = (ins/np.max(ins)*255).astype(np.int8)
    ref = (ref/np.mean(ref)*80).astype(np.int8)
    nr = (nr/np.mean(nr)*80).astype(np.int8)
    img = np.zeros((64,2048), dtype = (np.uint8))

    ind = np.where(rang == 0)[0]
    ins = np.delete(ins,list(ind))
    x = np.delete(x,list(ind))
    y = np.delete(y,list(ind))
    z = np.delete(z,list(ind))
    rang = np.delete(rang,list(ind))
    #print(np.arcsin(z/rang))
    u = (row_scale*(-((np.arcsin(z/rang)+fov_down)/0.784))).astype(np.int16)
    v = col_scale*(0.5*((np.arctan2(y,x)/3.141)+1))

    image = image_projection(img,u,v,ins)
    cv2.imwrite('./intensity_proj.png',image)
    image = image_projection(img,u,v,ref)
    cv2.imwrite('./reflectivity_proj.png',image)
    image = image_projection(img,u,v,nr)
    cv2.imwrite('./near_range_proj.png',image)
    return image
    # cv2.imshow('lol',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    


if __name__=="__main__":
    main()