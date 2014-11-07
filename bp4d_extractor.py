import numpy as np
import os, shutil
import csv
import fnmatch
import util
import pdb 

"""
This file provides methods to help extract appropriate images from the BP4D database
"""

### FILE EXTRACTION METHODS ###
def get_au_images(au):
    """
    Get au value for each frame for each user
    """
    au_path = "./data/bp4d/AU/Project/AU%d"%au
    csv_files = filter(lambda x: x.rsplit('.')[-1] == 'csv',  os.listdir(au_path))
    image_au_codes = dict()
    for csv in csv_files:
        meta, _ = csv.split('.')
        image_au_codes[meta] = get_frame_au_map( au_path + "/" + csv)
    return image_au_codes

def get_frame_au_map(csv_path):
    """
    Returns 2d numpy array where the first column is the frame number and the second column is the au value.
    """
    au_ratings = np.genfromtxt(csv_path)
    frame_numbers = np.where(au_ratings <=5)[0]

    frame_au = np.zeros([frame_numbers.shape[0], 2])

    frame_au[:,0] = frame_numbers # frame numbers
    frame_au[:,1] = au_ratings[frame_numbers] # au of frame
    return frame_au

def write_image_au(frame_au, out_path ='./data/au12/'):
    """
    Takes in the output of get_au_images and writes out the files to a new directory
    @params frame_au dict(string name : np.array()), np.array, first col = frame number, 2nd = au value
    """ 
    image_path = out_path + 'images'
    au_path = out_path + 'images'
    for fname, au_values in frame_au.iteritems():
        all_src_img_paths = get_src_image_paths(fname, au_vales)
        
        for src_path in all_src_img_paths:
            shutil.copyfile(src_path, out_path)
    
                
def get_src_image_paths(name, au_frames, src_path='/data/efros/byin/cs.binghamton/edu/\~gaic/Sequences(2D+3D)/'):
    """
    Returns a path to the corresponding image file based on the name.
    @params name
    @params au_frames np.array nx2 array; first column = index, 2nd column = au_value
    """
    components = name.split('_')
    gender = components[0][1]
    p_index = '0'+components[0][2:]
    task_id = components[1]
    
    img_path = "%s%s/T%s/"%(gender,p_index,task_id)
    all_paths = []
    for i in xrange(au_frames.shape[0]):
        img_file_name = "%04d.jpg"%au_frames[i,0]
        all_paths.append(src_path + img_path + img_file_name)
    return all_paths
