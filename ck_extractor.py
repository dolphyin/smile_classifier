import numpy as np
import os
import fnmatch
import util

"""
This file provides methods to help extract appropriate images from the CK++ database
"""

### FILE EXTRACTION METHODS ###
def get_au(f, au):
    """
    Returns the value of a specified AU in a given file. If the au does not exist, return -1.0 
    @params f file
    @params au int 
    @return float au value
    """
    for line in f:
        if not line.strip():
            continue
        tokens = line.split()
        if au == float(tokens[0]):
            return float(tokens[1])
    return -1.0 

def get_images(img_lst):
    """
    Returns a Nximg.shape np array of each image in img_lst
    """
    max_shape = (0,0)
    # intialize array for images 
    for i in xrange(len(img_lst)):
        prefix = img_lst[i]
        file_path = "./data/cohn-kanade-images/%s/%s/%s.png"%(prefix[0], prefix[1], '_'.join(prefix[0:3]))
        shape = util.read_image(file_path, True).shape
        max_shape = map(max, zip(*[max_shape, shape]))

    images = np.zeros([len(img_lst), max_shape[0], max_shape[1]])
   
    # get np array of each image
    for i in xrange(len(img_lst)):
        prefix = img_lst[i]
        file_path = "./data/cohn-kanade-images/%s/%s/%s.png"%(prefix[0], prefix[1], '_'.join(prefix[0:3]))
        img = util.read_image(file_path, True)
        # TODO: Change this later
        img = np.resize(img, max_shape)
        images[i,:,:] = img
    return images

def get_image(img_prefix):
    """
    Returns a the image for the given img_prefix
    """
    # intialize array for images 
    file_path = "./data/cohn-kanade-images/%s/%s/%s.png"%(img_prefix[0], img_prefix[1], '_'.join(img_prefix[0:3]))
    return util.read_image(file_path, True)

def get_landmark(img_prefix):
    """
    Returns a numpy array containg the landmarks of the specified img_prefix
    """
    file_path = "./data/Landmarks/%s/%s/%s_landmarks.txt"%(img_prefix[0], img_prefix[1], '_'.join(img_prefix[0:3]))
    return np.loadtxt(file_path)

### FILE NAMING METHODS ###
def get_image_prefixes(img_lst):
    """
    Returns the components of each image in img_lst required to reconstruct the paths. 
    @params: img_lst list[string] list of full path to image
    @return: list[list[str]] list of list of file directory
    """
    prefixes = []
    for img in img_lst:
        prefixes.append(img.split("_")[0:3])
    return prefixes

def get_image_name(img_prefix):
    return img_prefix.split("_")[0:3]

### DATA SUBSET ###
def get_data_subset(au, ck_path="./data", out_path="./data_subset"):
    """
    Gets a subset of the data and writes the images  to out_path/images
    and au values to out_path/FACS.
    @params au int au value to find 
    @params ck_path string path to ck++ database
    @params out_path string path to write out to
    """
    # get images of a given au
    image_paths = []
    for dirpath, dirnames, filenames in os.walk("%s/FACS"%ck_path):
        for filename in fnmatch.filter(filenames, "*.txt"):
            with open(dirpath + "/" + filename) as f:
                au_val = get_au(f, au)
                if au_val != -1.0:
                    image_paths.append((filename, au_val))

    # copy images and AU files into out_path
    image_out = "%s/images"%out_path
    au_out = "%s/FACS"%out_path
    for img_prefix, au in image_paths:
        image = get_image_name(img_prefix) 
        util.write_image(
    return image_paths
