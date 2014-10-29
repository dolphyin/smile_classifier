import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, fnmatch
import pdb

from scipy import io
from scipy import interpolate
from sklearn import svm 
from sklearn.decomposition import PCA

import util
import ck_extractor as extract

"""
Tips: Running out of memory, write out filtered images out to disk as you compute them

For dimensionality, sample images
For gabor filter variables: definitely need to include different orientations
                            - can try scale later if it works or not

Try for small dataset (10 images?)
    - see what works or what doesn't
"""

### KEYPOINT EXTRACTION ###
def get_keypoints(image_path, keypoint_path):
    # get list of image names
    image_names = os.listdir(image_path)

    img_size = util.read_image("%s/%s"%(image_path,image_names[0])).shape
    keypoint_size = io.loadmat("%s/%s"%(keypoint_path, image_names[0]), appendmat=True)['output']['pred'][0][0].shape

    results = np.zeros(len(image_names), dtype=[('file_name','a30'), ('image', '%s float32'%str(img_size)), ( 'keypoints', '%s float32'%str(keypoint_size))])
    for i in xrange(len(image_names)):
        name = image_names[i]
        img = util.read_image("%s/%s"%(image_path,image_names[i]))
        keypoints = io.loadmat("%s/%s"%(keypoint_path, image_names[i]), appendmat=True)['output']['pred'][0][0]
        results[i] = (name, img, keypoints)
    return results


### EXTRACTION METHODS ###
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
              
### GABOR FUNCTIONS ###
def get_gabor_training(images_kp, sizes=[17], num_orientations=8):
    """
    Take in a numpy array that contains images and keypoints and outputs a file with feature and class intensity.
    """ 
    #TODO: find way to get classes and intensity values

    # generate gabor filters to use
    filters = get_gabor_filters(sizes, num_orientations)

    # get features
    sample_image = images_kp[0]['image']
    sample_kp = images_kp[0]['keypoints']
    feature_size = len(get_gabor_features(sample_image, sample_kp, 2, filters))
    features = np.zeros([images_kp.shape[0], feature_size])
    for i in xrange(images_kp.shape[0]):
        img = images_kp[i]['image']
        kp = images_kp[i]['keypoints']
        features[i] = get_gabor_features(img, kp, 2, filters)
    return features


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

def keypoint_area(img, keypoints, rng):
    """
    Get the values around each keypoint in a box with dimension rngXrng around the keypoints
    Return a 1 dimensional vector
    """
    # TODO: handle images with color channel
    area_size = (2*rng)**2
    keypoint_vals = np.zeros([keypoints.shape[0], (2*rng)**2])
 
    # get interpolation functions for each channel
    interp = interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img)

    for i in xrange(keypoints.shape[0]):
        kx, ky = int(keypoints[i][0]), int(keypoints[i][1])
     
        interp_area = interp(np.arange(kx-rng, kx+rng), np.arange(ky-rng, ky+rng))

        keypoint_vals[i,:] = np.ravel(interp_area)
    return np.ravel(keypoint_vals)

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

### DIMENSTIONALITY REDUCTION ###
def reduce_PCA(features, n_components=162):
    """
    Reduces the dimensionality of features using PCA.
    """
    norm = util.normalize(features) 
    pca = PCA(n_components)
    reduced = pca.fit_transform(features)
    print('explained variance (first %d components): %.2f'%(n_components, sum(pca.explained_variance_ratio_)))
    return reduced

data = get_keypoints('./mini_data', './mini_data_out')
training = get_gabor_training(data)
reduced = reduce_PCA(training)
util.write_array('./train/train_small.txt', reduced)
