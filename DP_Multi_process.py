import numpy as np
from numpy.core.fromnumeric import shape
from scipy.ndimage.filters import convolve
import cv2
import math
import time 
from threading import Thread
from multiprocessing import Process, Manager

class SeamEnergyWithBackPointer():
    def __init__(self, energy, x_coordinate_in_previous_row=None):
        self.energy = energy
        self.x_coordinate_in_previous_row = x_coordinate_in_previous_row

def calculate_energy(img):
    filter_du = np.array([
        [1.0,2.0,1.0],
        [0.0,0.0,0.0],
        [-1.0,-2.0,-1.0]])
    filter_du= np.stack([filter_du]*3,axis=2)

    filter_dv = np.array([
        [1.0,0.0,-1.0],
        [2.0,0.0,-2.0],
        [1.0,0.0,-1.0]])
    filter_dv= np.stack([filter_dv]*3,axis=2)

    img = img.astype("float32")

    convolved = (np.absolute(convolve(img, filter_du))+ 
                            np.absolute(convolve(img, filter_dv)))

    energy_map = convolved.sum(axis=2)
    return energy_map

def caculate_cumulative(energy_img):
    seam_energies = []
    seam_energies.append([
        SeamEnergyWithBackPointer(pixel_energy)
        for pixel_energy in energy_img[0]])

    for y in range(1, len(energy_img)):
        
        pixel_energies_row = energy_img[y]
        seam_energies_row = []

        for x, pixel_energy in enumerate(pixel_energies_row):
            x_left = max(x - 1, 0)
            x_right = min(x + 1, len(pixel_energies_row) - 1)
            x_range = range(x_left, x_right + 1)
            
            min_parent_x = min(
                x_range,
                key=lambda x_i: seam_energies[y - 1][x_i].energy)

            min_seam_energy = SeamEnergyWithBackPointer(
                pixel_energy + seam_energies[y - 1][min_parent_x].energy,
                min_parent_x)

            seam_energies_row.append(min_seam_energy)

        seam_energies.append(seam_energies_row)
    return np.array(seam_energies, dtype=object)

def minimun_seam(seam_energies):
    

    min_seam_end_x = min(
            range(len(seam_energies[-1])),
            key=lambda x: seam_energies[-1][x].energy)
   
    seam = []
    seam_point_x = min_seam_end_x
    
    for y in range(len(seam_energies) - 1, -1, -1):
        seam.append([y,seam_point_x])
        
        seam_point_x =  seam_energies[y][seam_point_x].x_coordinate_in_previous_row

    seam.reverse()
    return np.array(seam, dtype=object)



def delete_seam(img,energy_img,matrix_cumulative,seam):
    r,c,_  = img.shape
    mask = np.ones((r, c), dtype=np.bool)
    
    for i in seam:
        mask[i[0],i[1]]=False
    mask_3 = np.stack([mask] * 3, axis=2)

    img = img[mask_3].reshape((r, c - 1, 3))
    energy_img = energy_img[mask].reshape((r, c - 1))
    matrix_cumulative = matrix_cumulative[mask].reshape((r, c - 1))

    return img, energy_img, matrix_cumulative

def seam_carving(img,number,return_img,name):


    for i in range(number):
        energy_img = calculate_energy(img)
        matrix_cumulative = caculate_cumulative(energy_img)
        seam = minimun_seam(matrix_cumulative)
        img,_,_ = delete_seam(img,energy_img,matrix_cumulative,seam)
    return_img[name] =img

def re_sofmax(lst,n_seam):
    sum_exp_lst = 0
    for i in lst:
        sum_exp_lst+=math.exp(-i)
    result = [int(n_seam*(0.05+0.75*math.exp(-x)/sum_exp_lst)) for x in lst]
    return result


def split_n_seam(img, n_seam,step):

    img_energy = calculate_energy(img)
    total_energy = np.sum(img_energy)
    #persent energy
    lst_p = [
        ((np.sum(img_energy[:,:step*1])      /total_energy)*100),
        ((np.sum(img_energy[:,step*1:step*2])/total_energy)*100),
        ((np.sum(img_energy[:,step*2:step*3])/total_energy)*100),
        ((np.sum(img_energy[:,step*3:step*4])/total_energy)*100),
        ((np.sum(img_energy[:,step*4:])      /total_energy)*100)]
    p = re_sofmax(lst_p,n_seam)
 
    return {"p0":p[0], "p1":p[1], "p2": p[2],"p3": p[3],"p4": p[4]}


if __name__=="__main__":
    path = "./images/lake.jpg"
    path_save = "./output/lake_DPMP.jpg"
    img = cv2.imread(path)
    #img =cv2.resize(img,(1600,900))

    n_seam = 100

    step = int(img.shape[1]/5)
    split_nseam = split_n_seam(img,n_seam,step)

    #print(split_nseam)
  
    img_crop0 = img[:,:(step*1),:]
    img_crop1 = img[:,(step*1):(step*2),:]
    img_crop2 = img[:,(step*2):(step*3),:]
    img_crop3 = img[:,(step*3):(step*4),:]
    img_crop4 = img[:,(step*4):,:]

    return_img = Manager().dict()

    jobs=[]

    start=time.time()

    t0 = Process(target=seam_carving, args=(img_crop0,split_nseam["p0"],return_img,"step0"))
    t1 = Process(target=seam_carving, args=(img_crop1,split_nseam["p1"],return_img,"step1"))
    t2 = Process(target=seam_carving, args=(img_crop2,split_nseam["p2"],return_img,"step2"))
    t3 = Process(target=seam_carving, args=(img_crop3,split_nseam["p3"],return_img,"step3"))
    t4 = Process(target=seam_carving, args=(img_crop4,split_nseam["p4"],return_img,"step4"))
    jobs.append(t0)
    jobs.append(t1)
    jobs.append(t2)
    jobs.append(t3)
    jobs.append(t4)
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    stop = time.time()
    print(stop - start)


    for proc in jobs:
        proc.join()
    
      
    img_result = np.concatenate((return_img["step0"],
                                    return_img["step1"],
                                    return_img["step2"],
                                    return_img["step3"],
                                    return_img["step4"]),
                                    axis=1)
  
    cv2.imshow("img_orinal",img)
    cv2.imshow("img_resizesd",img_result)
    cv2.waitKey(0)
   
    cv2.imwrite(path_save,img_result)
   