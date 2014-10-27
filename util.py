from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy

"""
This file contains a bunch of helper methods to read, write, and display images
""" 

def read_image(path, grayscale=False):
    """
    Reads an image from the given path and returns np array
    """ 
    return cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def write_image(path, img):
    """
    Writes img to a file located at path
    """
    misc.imsave(path, img) 

def display_image(data,points=None):
    """
    Utility method to display image data
    """
    tmp = np.double(copy.deepcopy(data))
    if data.dtype == float:
        tmp = normalize(tmp)
    plt.imshow(tmp)
    if points!= None:
        plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

def normalize(data):
    """
    Normalizes the data
    """
    return (data - np.amin(data))/(np.amax(data)-np.amin(data))

def overlay_SIFT(img, keypoints, dest=None):
    """
    Returns image with keypoints overlayed on image
    """
    return cv2.drawKeypoints(img,keypoints)
