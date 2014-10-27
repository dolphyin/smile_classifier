import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, fnmatch
import pdb

from sklearn import svm 
from skimage.filter import gabor_kernel
from sklearn.decomposition import PCA

import util
import data_extractor as extract

"""
Tips: Running out of memory, write out filtered images out to disk as you compute them

For dimensionality, sample images
For gabor filter variables: definitely need to include different orientations
                            - can try scale later if it works or not

Try for small dataset (10 images?)
    - see what works or what doesn't
"""


### EXTRACTION METHODS ###

def get_SIFT_features(image_path, mask=None):
    """
    Takes in a NxHxW numpy array and outputs a Nx??? numpy array of the keypoints
    detected by SIFT.

    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT()
    mask = cv2.convertScaleAbs(mask)
    kp, des = sift.detectAndCompute(gray, mask)
    return kp, des 

def create_mask(img, landmarks, rng):
    """
    Finds all non-zero points in a mask and expands the range of max around these points by
    rng in each direction
    
    TODO: Add checks if range goes beyond image range
    """
    # set intial mask points
    mask = np.zeros(img.shape)
    landmarks = np.int_(landmarks)
    mask[landmarks[:,0], landmarks[:,1]] = 1 
    points = zip(*np.where(mask!=0))

    for p in points:
        for x in xrange(-rng, rng, 1):
            for y in xrange(-rng, rng, 1):
                x_point = p[0] + x
                y_point = p[1] + y
                mask[x_point,y_point] = 1
    return mask
              
### Get Data Methods ###
def get_SIFT_training(prefixes):
    sift_des = []

    for prefix in prefixes:
        img_path = "./data/cohn-kanade-images/%s/%s/%s.png"%(prefix[0], prefix[1], '_'.join(prefix[0:3]))

        img =  extract.get_image(prefix)

        landmarks = extract.get_landmark(prefix)
        landmarks = landmarks.astype(int)

        new = np.zeros(landmarks.shape, dtype=int)
        new[:,0] = landmarks[:,1]
        new[:,1] = landmarks[:,0]

        mask = create_mask(img,new, 12)
        kp, des = get_SIFT_features(img_path, mask)
        sift_des.append(des)

        #io.display_image(io.overlay_SIFT(img,kp, None))
        #io.display_image(img, mask)
    return sift_des

### GABOR FUNCTIONS ###
def get_gabor_training(prefixes):
    
    landmarks, prefixes = extract.get_image_landmarks(prefixes) 

    # generate gabor filters to use
    filters = get_gabor_filters([17], 2)
    images = extract.get_images(prefixes)

    # get features
    feature_size = len(get_gabor_features(images[0], landmarks[0], 2, filters))
    features = np.zeros([images.shape[0], feature_size])
    for i in xrange(images.shape[0]):
        f = get_gabor_features(images[i], landmarks[i], 2, filters)
        features[i] = get_gabor_features(images[i], landmarks[i], 2, filters)
    return features

def keypoint_area(img, keypoints, rng):
    """
    Get the values at the keypoints by interpolating the values
    """
    keypoint_vals = np.zeros([keypoints.shape[0], (2*rng)**2])
    for i in xrange(keypoints.shape[0]):
        kx, ky = int(keypoints[i][0]), int(keypoints[i][1])
        keypoint_vals[i,:] = np.ravel(img[kx-rng:kx+rng, ky-rng:ky+rng])
    return np.ravel(keypoint_vals)

def get_gabor_features(img, keypoints, rng, filters):
    """
    Return a flattened array of pixel values around the keypoints
    """
    mask = create_mask(img, keypoints, rng)

    num_filters = len(filters.keys())
    num_features = keypoint_area(img, keypoints, rng).size
    feature_dim = [num_filters, num_features ]  # num_filters x pixels in local square regions around keypoints 
    features = np.zeros(feature_dim) 
    for i, f in enumerate(filters.values()):
        convolved = cv2.filter2D(img, cv2.CV_8UC3,f)
        features[i,:] = keypoint_area(convolved, keypoints, rng)
    return np.ravel(features)

def get_gabor_filters(ksizes, num_orientations):
    """
    Creates a dictionary of gabor filters of various sizes and orientations
    @params ksizes list[int] size of box filter
    @params num_orientations int number of equally divided orientation of filters between 0 and 180 degrees.
    @return dict{size} = [cv2.gaborKernel]
    """
    filters = dict()
    for size in ksizes:
        for i in range(num_orientations):
            theta = np.pi/num_orientations * i
            filters[(size, i)] = cv2.getGaborKernel((size,size), 4, theta, 10.0, 0.5, 0)
    return filters 

if __name__ == "__main__":
    # get all prefixes to files that contain AU12 
    #prefixes = extract.get_au_prefixes(12)
    #landmarks, prefixes = extract.get_image_landmarks(prefixes) # get landmarks and prefixes that have landmarks
    pass
    # generate gabor filters to use
    #filters = get_gabor_filters([17,23,33,46,65], 8)

#get valid files
prefixes = extract.get_au_prefixes(12)
landmarks, prefixes = extract.get_image_landmarks(prefixes) 

# generate gabor filters to use
filters = get_gabor_filters([17], 2)
images = extract.get_images(prefixes)

# get features
feature_size = len(get_gabor_features(images[0], landmarks[0], 2, filters))
features = np.zeros([images.shape[0], feature_size])
for i in xrange(images.shape[0]):
    f = get_gabor_features(images[i], landmarks[i], 2, filters)
    features[i] = get_gabor_features(images[i], landmarks[i], 2, filters)
