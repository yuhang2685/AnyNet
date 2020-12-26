import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
# YH: Specify it is from the same directory that import below modules 
from . import preprocess
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np

''' 
YH: This file defines a class 'myImageFloder' 
    containing method for loading an image and preprocessing
'''

# YH: The set of image file extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


# YH: Determine if the file is an image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# YH: Converts the file into a PIL 'RGB' format
def default_loader(path):
    return Image.open(path).convert('RGB')


# YH: Load a Disparity map in the PFM format
def disparity_loader(path):
    return rp.readPFM(path)

'''
YH: A map-style dataset is one that implements the "__getitem__()" and "__len__()" protocols, 
    and represents a map from (possibly non-integral) indices/keys to data samples.
    For example, such a dataset, when accessed with dataset[idx], 
    could read the idx-th image and its corresponding label from a folder on the disk.
'''
# YH: The class argument indicates the class instance is a "data.Dataset".
class myImageFloder(data.Dataset):

    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        # YH: Functional programming to define the property as a function.
        self.loader = loader
        self.dploader = dploader
        self.training = training

    '''
    YH: This method "__getitem__" is required for being a "data.Dataset".
        Pre and post double underscore prevents others use them as variable names (runtime errors). 
        The method by index gets a particular items such as Left, Right, and Disparity map from the dataset.
    '''
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        # YH: Note it only has Left disparity, which is actually the Ground truth.
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        # YH: Sort dataL in ascending order
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        # YH: If it is in the training procedure
        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            
            # YH: Random crop a 256/512 piece from both images.
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            # YH: Pick the crop corresponding Disparity maps
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            # YH: Assign varialbe "processed" as a composed transformations (functions).
            processed = preprocess.get_transform(augment=False)
            # YH: Apply "processed" procedures over images (Starting from "transforms.ToTensor()"").
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            # YH: If it is not training, crop the right bottom part of 960/544 piece from both images.
            left_img = left_img.crop((w - 960, h - 544, w, h))
            right_img = right_img.crop((w - 960, h - 544, w, h))
            
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL


    def __len__(self):
        return len(self.left)
