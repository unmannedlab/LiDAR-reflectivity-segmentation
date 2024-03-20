import cv2
import open3d
import numpy as np
import math

np.random.seed(12)
pal= np.random.randint(0, 256, (256, 3), dtype=np.uint8)



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

def image_translate(image):
    num_pixels = 300
    width = image.shape[1]
    translated_image = np.zeros_like(image)
    print(translated_image.shape)
    translated_image[:, :width - num_pixels] = image[:, num_pixels:]
    translated_image[:, width - num_pixels:] = image[:, :num_pixels]
    return translated_image

def image_projection(img,u,v,ins):
    for i in range(len(u)):
        img[u[i]][int(v[i])] = ins[i]
    #img = pal[img]
    # temp = img.copy()
    # image = np.zeros((img.shape[0],img.shape[1],3))
    # for k, v in  color_palette.items():
    #         print(v['color'])
    #         image[temp == k, :] = v["color"]
    # print(image.shape)
    image = cv2.flip(img,1)
    image = image_translate(image)
    #return img
    cv2.imwrite('./nr_584.png',image)
    #cv2.imshow('lol',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def load_npy(npy_path):
    obj = np.load(npy_path,allow_pickle=True)
    return obj

def main(points=None):
    points = load_bin('./000584.bin')
    labels = np.fromfile('./000584.label',dtype = np.int32,count = -1)
    #points = load_npy('/home/usl/Desktop/random/000000_cust1.npy')
    #print(points.shape)
    #x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,3]*65535).astype(np.int16)
    x,y,z,ins= points[:,0],points[:,1],points[:,2],(points[:,5])
    fov_up = 0.392
    fov_down = -0.392
    rang = np.sqrt(x**2+y**2+z**2)
    row_scale = 64
    col_scale = 2047
    ins = (ins/np.mean(ins)*80).astype(np.int8)
    #labels = (labels/np.mean(labels)*255).astype(np.int8)
    img = np.zeros((64,2048), dtype = (np.uint8))

    # ind = np.where(rang == 0)[0]
    # ins = np.delete(ins,list(ind))
    # x = np.delete(x,list(ind))
    # y = np.delete(y,list(ind))
    # z = np.delete(z,list(ind))
    # rang = np.delete(rang,list(ind))
    #print(np.arcsin(z/rang))
    u = (row_scale*(-((np.arcsin(z/rang)+fov_down)/0.784))).astype(np.int16)
    v = col_scale*(0.5*((np.arctan2(y,x)/3.141)+1))

    image = image_projection(img,u,v,ins)
    return image
    # cv2.imshow('lol',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    


if __name__=="__main__":
    main()