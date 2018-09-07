# -*-coding:UTF-8-*-
import os
import scipy.io
import numpy as np
import glob
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import Mytransforms
import sys
import matplotlib.pyplot as plt

def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    #glob() 함수는 경로에 대응되는 모든 파일 및 디렉터리의 리스트를 반환
    image_arr = np.array(glob.glob(os.path.join(root_dir, 'images/*.jpg')))
    #print(__file__)
    #print('image_arr\n',image_arr,image_arr.shape)
    #print(s) for s in image_arr
    
    image_nums_arr = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_arr])
    #print('image_nums_arr\n',image_nums_arr,type(image_nums_arr),image_nums_arr.size)
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    #print('sorted_image_arr\n',sorted_image_arr,'\n',type(image_nums_arr),sorted_image_arr.shape)
    #print(sorted_image_arr.tolist(),type(sorted_image_arr.tolist()),len(sorted_image_arr.tolist()))
    #sys.exit(1)
    #tolist()  * to list format
    return sorted_image_arr.tolist()
    
def read_sangi_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    image_arr = np.array(glob.glob(os.path.join(root_dir, 'sangi/ASR_MAIN/*.jpg')))
    #print('image_arr\n',image_arr,image_arr.shape)
    #original images
    image_nums_arr = np.array([float(s.rsplit('.')[-2][-3:]) for s in image_arr])
    
    #for crop images
    #image_nums_arr = np.array([float(s.rsplit('.')[-2][-5:]) for s in image_arr])
    
    #print('image_nums_arr\n',image_nums_arr,type(image_nums_arr),image_nums_arr.size)
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    #print('sorted_image_arr\n',sorted_image_arr,'\n',type(image_nums_arr),sorted_image_arr.shape)
    
    return sorted_image_arr.tolist()
    
    
def read_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'lsp' or 'lspet'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints'] # read mat file
    # lspnet (14,3,10000)
    if mode == 'lspet':
        lms = mat_arr.transpose([2, 1, 0])
        kpts = mat_arr.transpose([2, 0, 1]).tolist()
    # lsp (3,14,2000)
    if mode == 'lsp':
		# One person has 14 joints, so shape of mat_arr[2] is (14,2000) 
        mat_arr[2] = np.logical_not(mat_arr[2]) # 0->1 1->0
        lms = mat_arr.transpose([2, 0, 1]) # http://superelement.tistory.com/18 transpose axis exp
        kpts = mat_arr.transpose([2, 1, 0]).tolist() # can't know array shape, if u wanna know shape of 'kpts' change array to np.array
        #print('\n',kpts,'\n')
        # tolist() returns the array as a (possibly nested) list.
        

    centers = []
    scales = []
    for idx in range(lms.shape[0]):# lms.shape[0]=2000 [1]=3 [2]=14
        im = Image.open(img_list[idx])
        
        w = im.size[0]
        h = im.size[1]
        #print('\n',lms[0][1][:])
        # lsp and lspet dataset doesn't exist groundtruth of center points
        #center of all x joints
        center_x = (lms[idx][0][lms[idx][0] < w].max() +      # lms[idx][0][:] - all of x joint points in idx frame  
                    lms[idx][0][lms[idx][0] > 0].min()) / 2    # max and min x of 14 joints is picked and find center x
        '''
        print('w\n',w)
        print(lms.shape[idx][0])
        print(lms[idx][0][lms[idx][0] < w].max())
        
		'''
       
        
        #center of all y joints 
        center_y = (lms[idx][1][lms[idx][1] < h].max() +      # lms[idx][1][:] - all of y joint points in idx frame
                    lms[idx][1][lms[idx][1] > 0].min()) / 2    # max and min y of 14 joints is picked and find center y
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0 # '+4'?
        scales.append(scale)
	
	# kpts - all of info of joints
	# centers - center of joints
	# scales - ratio of joints area to each frame size 
	
    return kpts, centers, scales
    
def read_sangi_mat_file(mode, root_dir, img_list):
    """
        get the groundtruth

        mode (str): 'lsp' or 'lspet'
        return: three list: key_points list , centers list and scales list

        Notice:
            lsp_dataset differ from lspet dataset
    """
    mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints.mat'))['joints']
    # lspnet (14,3,10000)
    if mode == 'lspet':
        lms = mat_arr.transpose([2, 1, 0])
        kpts = mat_arr.transpose([2, 0, 1]).tolist()
    # lsp (3,14,2000)
    if mode == 'lsp':
        mat_arr[2] = np.logical_not(mat_arr[2])
        lms = mat_arr.transpose([2, 0, 1])
        kpts = mat_arr.transpose([2, 1, 0]).tolist()

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]
        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return kpts, centers, scales


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

number=0

class LSP_Data(data.Dataset):
    """
        Args:
            root_dir (str): the path of train_val dateset.
            stride (float): default = 8
            transformer (Mytransforms): expand dataset.
        Notice:
            you have to change code to fit your own dataset except LSP

    """
    
    def __init__(self, mode, root_dir, stride, transformer=None):
       
        #self.img_list = read_sangi_file(root_dir)
        
        self.img_list = read_data_file(root_dir) 
        
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        
        self.stride = stride 
        
        self.transformer = transformer
        self.sigma = 3.0
        


    def __getitem__(self, index):
        global number
        number=number+1
        
        img_path = self.img_list[index]
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        
        
        kpt = self.kpt_list[index] # kpt is maybe key point namely joint
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        img, kpt, center = self.transformer(img, kpt, center, scale)
        height, width, _ = img.shape
        heatmap = np.zeros((int(height / self.stride), int(width / self.stride), int(len(kpt) + 1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=height / self.stride, size_w=width / self.stride, center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background

        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                     [256.0, 256.0, 256.0])
        heatmap = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        return img, heatmap, centermap, img_path
        #return img, heatmap, centermap

    def __len__(self):
        #print('len')
        return len(self.img_list) #return # of images


