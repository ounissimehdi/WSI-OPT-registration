import numpy as np
import scipy as sp
import pandas as pd

# Import transforms
from transforms import CompositeTransform
from transforms import AffineTransform
from transforms import Rigid2DTransform
from transforms import Rotate2DTransform
from transforms import TranslationTransform
from transforms import ScalingTransform
import transforms
import scipy.ndimage.interpolation

# Import optimizers
from optimizers import GradientDescentOptimizer

# Import generators and filters
import filters

# Import registration framework
from register import Register

# Import misc
import time
import os
import cv2
import shutil

from glob import glob
from natsort import natsorted
from PIL import Image
from tqdm import trange

import argparse



def transform(pnts, tf_param, dim=2):
    # DIM = 2 for 2D, DIM = 3 for 3D
    m = np.transpose(tf_param[:dim*dim].reshape((dim, dim)))
    t = tf_param[dim*dim:]

    return pnts.dot(m) + t

def compute_grid(tfs_param, Out, out_spacing, old_grid, memory):

    if not(memory):

        linspaces = [np.linspace(0, Out.shape[i]*out_spacing[i], Out.shape[i], endpoint=False) for i in range(Out.ndim)]

        grid = np.array(np.meshgrid(*linspaces,indexing='ij'))

        grid = grid.reshape((Out.ndim, np.prod(Out.shape)))
        grid = np.moveaxis(grid, 0, 1)

        t = ((np.array(Out.shape)-1) * spacing) * 0.5
        
        # for i in range(len(tfs_param)):
        grid = grid - t
        grid = transform(grid, tfs_param)
        grid = grid + t

        grid_transformed = grid

        grid_transformed[:, :] = grid_transformed[:, :] * (1.0 / out_spacing[:])
        
        grid_transformed = np.moveaxis(grid_transformed, 0, 1)
        grid_transformed = grid_transformed.reshape((Out.ndim,) + Out.shape)

        return grid_transformed, grid
    
    else:
        t = ((np.array(Out.shape)-1) * spacing) * 0.5
        
        # for i in range(len(tfs_param)):
        old_grid = old_grid - t
        old_grid = transform(old_grid, tfs_param)
        old_grid = old_grid + t

        grid_transformed = old_grid

        grid_transformed[:, :] = grid_transformed[:, :] * (1.0 / out_spacing[:])
        
        grid_transformed = np.moveaxis(grid_transformed, 0, 1)
        grid_transformed = grid_transformed.reshape((Out.ndim,) + Out.shape)
        
        return grid_transformed, old_grid

        

def grid_warp_img(grid_transformed, In, Out, bg_value, mode='nearest'):

    if mode == 'spline':
        scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, cval = bg_value)
    elif mode == 'linear':
        scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=1, cval = bg_value)
    elif mode == 'nearest':
        scipy.ndimage.interpolation.map_coordinates(In, coordinates=grid_transformed, output=Out, order=0, cval = bg_value)

def compute_tf(ref_gray, flo_gray):
    np.random.seed(1000)

    ref_gray = filters.normalize(ref_gray, 0.0, None)           # x - x_min / x_max - x_min
    flo_gray = filters.normalize(flo_gray, 0.0, None)
 
    weights1 = np.ones(ref_gray.shape)                     
    mask1 = np.ones(ref_gray.shape, 'bool')
    weights2 = np.ones(flo_gray.shape)
    mask2 = np.ones(flo_gray.shape, 'bool')

    # Initialize registration framework for 2d images
    reg = Register(2)

    reg.set_report_freq(param_report_freq)                  
    reg.set_alpha_levels(alpha_levels)                      
    reg.set_reference_image(ref_gray)
    reg.set_reference_mask(mask1)
    reg.set_reference_weights(weights1)

    reg.set_floating_image(flo_gray)
    reg.set_floating_mask(mask2)
    reg.set_floating_weights(weights2)

    # Setup the Gaussian pyramid resolution levels
    
    reg.add_pyramid_level(4, 5.0)                                  
    reg.add_pyramid_level(2, 3.0)
    reg.add_pyramid_level(1, 0.0)

    # Learning-rate / Step lengths [[start1, end1], [start2, end2] ...] (for each pyramid level)
    step_lengths = np.array([[1., 1.], [1., 1.], [1., 1e-1]]) * 1e-1           

    # Create the transform and add it to the registration framework (switch between affine/rigid transforms by commenting/uncommenting)
    # Affine
    #1.0/diag, 1.0/diag, 1.0/diag, 1.0/diag, 1.0, 1.0
    reg.add_initial_transform(AffineTransform(2), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))   # add_initial_transform(transform, param_scaling)
    # Rigid 2D
    # reg.add_initial_transform(Rigid2DTransform(2), np.array([1.0/diag, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.001)                    
    reg.set_sampling_fraction(param_sampling_fraction)            
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('adam')

    # Create output directory
    directory = os.path.dirname('./test_images/output/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Start the pre-processing
    reg.initialize('./test_images/output/')
    
    # Control the formatting of numpy
    np.set_printoptions(suppress=True, linewidth=200)

    # Start the registration
    reg.run()

    # Retrieve the transformation results
    (transform, value) = reg.get_output(0)

    # Transformation matrix
    tf_matrix = transform.get_params()

    return tf_matrix

def main_registration(image_id):

    source_folder = os.path.join('..', 'HE','clean_dataset')

    output_path = os.path.join('..', 'HE','PAR_images_scale_'+str(scale))                #  path for input images     
    os.makedirs(output_path, exist_ok = True)        

    files = natsorted(glob(os.path.join(source_folder, '*.tif')))

    list_tf = []

    start_time = time.time()

    ref_im = cv2.imread(files[image_id-1])
    flo_im = cv2.imread(files[image_id])


    scale_percent = scale # percent of original size
    width = int(ref_im.shape[1] * scale_percent / 100)
    height = int(ref_im.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    ref_im = cv2.resize(ref_im, dim, interpolation = cv2.INTER_LINEAR)
    flo_im = cv2.resize(flo_im, dim, interpolation = cv2.INTER_LINEAR)

    ref_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    flo_gray = cv2.cvtColor(flo_im, cv2.COLOR_BGR2GRAY)

    tf_matrix = compute_tf(ref_gray, flo_gray)

    tmp_list_tf = [files[image_id], str(round(time.time()-start_time, 4))]

    for elm in tf_matrix:tmp_list_tf.append(elm)
    list_tf.append(tmp_list_tf)

    df_tfs = pd.DataFrame(list_tf, columns =['filename', 'time', 'xx', 'xy', 'yx', 'yy', 'X', 'Y'])
    df_tfs.to_csv(os.path.join(output_path, str(image_id)+'.csv'), index=False)

    
    # return df_tfs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-img_id" ,"--image_id", help="image to be registered",type=int)
    args = parser.parse_args()
    
    # Getting the acquisition ID
    image_id = args.image_id
    scale = 10               # percent of registered images size
    scale_percent = 10*scale # percent of original size

    # Registration Parameters
    alpha_levels = 7
    
    # Pixel-size
    spacing = np.array([1.0, 1.0])
    
    # Run the symmetric version of the registration framework
    symmetric_measure = True
    
    # Use the squared Euclidean distance
    squared_measure = False

    # The number of iterations
    param_iterations = 500
    
    # The fraction of the points to sample randomly (0.0-1.0)
    param_sampling_fraction = 0.1
    
    # Number of iterations between each printed output (with current distance/gradient/parameters)
    param_report_freq = 100

    start_time = time.time()
    df_tfs = main_registration(image_id)
    end_time = time.time()

    # print("Elapsed time: " + str((end_time-start_time)))

