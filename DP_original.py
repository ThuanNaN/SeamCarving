import numpy as np
from scipy.ndimage.filters import convolve
import cv2
import time 


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

def seam_carving(img,number):

    for _ in range(number):
        energy_img = calculate_energy(img)
        matrix_cumulative = caculate_cumulative(energy_img)
        seam = minimun_seam(matrix_cumulative)
        img,_,_ = delete_seam(img,energy_img,matrix_cumulative,seam)
    return img
        
if __name__ == "__main__":
    path = "./images/lake.jpg"
    path_save = "./output/lake_DP.jpg"

    img = cv2.imread(path)
    #img =cv2.resize(img,(1600,900))
    n = 200

    t=time.time()
    img_result = seam_carving(img,n)
    print("Total time: ",time.time()-t)

    cv2.imshow("Image Original",img)
    cv2.imshow("Result",img_result)
    cv2.waitKey(0)
    
    #cv2.imwrite(path_save,img_result)