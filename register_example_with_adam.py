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


def main(im1_path, im2_path, file1, file2, ref_gray, flo_gray, ref_im, flo_im):
    np.random.seed(1000)
    # if len(sys.argv) < 3:
    #     print('register_example.py: Too few parameters. Give the path to two gray-scale image files.')
    #     print('Example: python2 register_example.py reference_image floating_image')
    #     return False



    # Save copies of original images
    ref_im_orig = ref_gray.copy()

    ref_gray = filters.normalize(ref_gray, 0.0, None)           # x - x_min / x_max - x_min
    flo_gray = filters.normalize(flo_gray, 0.0, None)
    
    diag = 0.5 * (transforms.image_diagonal(ref_gray, spacing) + transforms.image_diagonal(flo_gray, spacing))          

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

    (transform, value) = reg.get_output(0)

    print(transform)
    print(np.shape(transform))


    ### Warp final image
    c = transforms.make_image_centered_transform(transform, ref_gray, flo_gray, spacing, spacing)

    # Transformation matrix
    tf_matrix = transform.get_params()

    # Print out transformation parameters
    # print('Transformation parameters:')
    # list_tf = [im2_path]
    # for elm in tf_matrix:list_tf.append(elm)

    # df = pd.DataFrame([list_tf], columns =['filename','xx', 'xy', 'yx', 'yy', 'X', 'Y'])
    # df.to_csv(os.path.join("transformation_matrix_10", os.path.basename(im2_path).split('.')[0]+'.csv'), index=False)


    # Create the output image
    flo_im_warped_B = np.zeros(ref_im_orig.shape)
    flo_im_warped_G = np.zeros(ref_im_orig.shape)
    flo_im_warped_R = np.zeros(ref_im_orig.shape)
    flo_im_warped_B.fill(255)
    flo_im_warped_G.fill(255)
    flo_im_warped_R.fill(255)


    # Transform the floating image into the reference image space by applying transformation 'c'
    flo_im_B = flo_im[..., 0]
    flo_im_G = flo_im[..., 1]
    flo_im_R = flo_im[..., 2]

    c.warp(In = flo_im_B, Out = flo_im_warped_B, in_spacing=spacing, out_spacing=spacing, mode='nearest', bg_value = 255.0)
    c.warp(In = flo_im_G, Out = flo_im_warped_G, in_spacing=spacing, out_spacing=spacing, mode='nearest', bg_value = 255.0)
    c.warp(In = flo_im_R, Out = flo_im_warped_R, in_spacing=spacing, out_spacing=spacing, mode='nearest', bg_value = 255.0)

    flo_im_warped = np.stack((flo_im_warped_B, flo_im_warped_G, flo_im_warped_R), axis = -1)
    flo_im_warped = flo_im_warped.astype(np.uint8)  

    # print(im2_path)

    cv2.imwrite(im2_path, flo_im_warped)                # Save the registered image 

    # Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im-flo_im_warped)
    err = np.mean(D1)
    print("Err: %f" % err)
    diff_dir = "diff"                   #  dir to save the difference between reference image and floating image
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
    cv2.imwrite(os.path.join(diff_dir, 'diff_{}_{}.png'.format(file1.split('.')[0], file2.split('.')[0])), D1)
    return True, tf_matrix, err


def main_registration():
    scales = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for scale in scales:
        os.makedirs(os.path.join('transformation_matrix'), exist_ok = True)
        source_folder = os.path.join('..', 'sub_img_to_reg')   
        destination_folder = os.path.join('images_'+str(scale))
        os.makedirs(destination_folder, exist_ok = True)
        list_tf = []

        # Deleting existing files
        for file_name in os.listdir(destination_folder):
            # construct full file path
            file = os.path.join(destination_folder, file_name)
            if os.path.isfile(file): os.remove(file)

        # fetch all files
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = os.path.join(source_folder, file_name)
            destination = os.path.join(destination_folder, file_name)
            # copy only files
            if os.path.isfile(source): shutil.copy(source, destination)

        path = os.path.join('images_'+str(scale))                #  path for input images               
        files = sorted(os.listdir(path), key = lambda x: int(x[:-4]))
        # print(files)
        
        for file1, file2 in zip(files, files[1:]):
            start_time = time.time()
            ## Track registration progress
            print("First input image:", file1)
            print("Second input image:", file2)
            print("\n")
            im1_path = os.path.join(path, file1)
            ## Read in ith image
            if file2 is not None:
                im2_path = os.path.join(path, file2)

            ref_im_path = im1_path
            flo_im_path = im2_path
            ref_im = cv2.imread(ref_im_path)
            flo_im = cv2.imread(flo_im_path)

            if file1 == files[0] and file2== files[1]:
                scale_percent = scale # percent of original size
                width = int(ref_im.shape[1] * scale_percent / 100)
                height = int(ref_im.shape[0] * scale_percent / 100)
                dim = (width, height)
                
                # resize image
                ref_im = cv2.resize(ref_im, dim, interpolation = cv2.INTER_LINEAR)
            flo_im = cv2.resize(flo_im, dim, interpolation = cv2.INTER_LINEAR)

            ref_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
            flo_gray = cv2.cvtColor(flo_im, cv2.COLOR_BGR2GRAY)

            res, tf_matrix, err = main(im1_path, im2_path, file1, file2, ref_gray, flo_gray, ref_im, flo_im)

            tmp_list_tf = [im2_path, str(round(time.time()-start_time, 4)), err]
            for elm in tf_matrix:tmp_list_tf.append(elm)
            list_tf.append(tmp_list_tf)

            df = pd.DataFrame(list_tf, columns =['filename', 'time', 'error', 'xx', 'xy', 'yx', 'yy', 'X', 'Y'])
            df.to_csv(os.path.join("transformation_matrix", str(scale)+'.csv'), index=False)

            ## Keep track of saved images
            print("Saved image:", file2)
            print("Elapsed time: " + str((time.time()-start_time)))
            print("________")
    return res



if __name__ == '__main__':
    start_time = time.time()
    res = main_registration()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
