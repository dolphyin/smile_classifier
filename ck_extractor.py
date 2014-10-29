import numpy as np
import os
import fnmatch
import util

"""
This file provides methods to help extract appropriate images from the CK++ database
"""

### FILE EXTRACTION METHODS ###
def get_au_prefixes(au):
    """
    Gets a list of images names that contain the AU.
    """
    image_paths = []
    for dirpath, dirnames, filenames in os.walk("./data/FACS"):
        for filename in fnmatch.filter(filenames, "*.txt"):
            with open(dirpath + "/" + filename) as f:
                if contains_au(f, au):
                    image_paths.append(filename)
    return get_image_prefix(image_paths)

def contains_au(f, au):
    """
    Returns whether or not a file f contains an AU
    @params f txt file
    @au int action unit code 
    @return boolean if action unit found
    """
    for line in f:
        if not line.strip():
            continue 
        if au == float(line.split()[0]):
            return True
    return False

### FILE NAMING METHODS ###
def get_image_prefix(img_lst):
    """
    Returns the components of each image in img_lst required to reconstruct the paths. 
    @params: img_lst list[string] list of full path to image
    @return: list[list[str]] list of list of file directory
    """
    prefixes = []
    for img in img_lst:
        prefixes.append(img.split("_")[0:3])
    return prefixes

def get_image_landmarks(img_prefix):
    """
    Returns a np.array Nx68x2 of the landmarks of all images in img_lst.
    Also returns an updated list of valid img_prefixes
    file was not found.
    """
    landmarks = np.zeros([len(img_prefix), 68, 2], dtype=np.double)
    not_found = [] 
    for i in xrange(len(img_prefix)):
        prefix = img_prefix[i]
        file_path = "./data/Landmarks/%s/%s/%s_landmarks.txt"%(prefix[0], prefix[1], '_'.join(prefix[0:3]))
        try:
            # catch cases where AU files have no corresponding landmarks
            landmarks[i, :, :] = np.loadtxt(file_path)
        except IOError:
            not_found.append(i)
            continue
    for i in not_found:
        del img_prefix[i]
        np.delete(landmarks, i, 0) 
    return landmarks, img_prefix 

def get_images(img_lst):
    """
    Returns a Nximg np array of each image in img_lst
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
    file_path = "./data/Landmarks/%s/%s/%s_landmarks.txt"%(img_prefix[0], img_prefix[1], '_'.join(img_prefix[0:3]))
    return np.loadtxt(file_path)


