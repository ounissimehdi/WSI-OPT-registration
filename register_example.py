import numpy as np
import scipy as sp
import cv2

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

# Registration Parameters
alpha_levels = 7
# Pixel-size
spacing = np.array([1.0, 1.0])
# Run the symmetric version of the registration framework
symmetric_measure = True
# Use the squared Euclidean distance
squared_measure = False

# The number of iterations
# param_iterations = 500
param_iterations = 500
# The fraction of the points to sample randomly (0.0-1.0)
param_sampling_fraction = 0.1
# Number of iterations between each printed output (with current distance/gradient/parameters)
param_report_freq = 50


def main():
    np.random.seed(1000)
    
    # if len(sys.argv) < 3:
    #     print('register_example.py: Too few parameters. Give the path to two gray-scale image files.')
    #     print('Example: python2 register_example.py reference_image floating_image')
    #     return False

    # ref_im_path = sys.argv[1]
    # flo_im_path = sys.argv[2]

    ref_im_path = "C:/Users/lhcez/Desktop/Code/py_alpha_amd_release/images/9.tif"
    flo_im_path = "C:/Users/lhcez/Desktop/Code/py_alpha_amd_release/images/10.tif"
    ref_im = cv2.imread(ref_im_path)
    flo_im = cv2.imread(flo_im_path)
    ref_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    flo_gray = cv2.cvtColor(flo_im, cv2.COLOR_BGR2GRAY)
    # Save copies of original images
    ref_im_orig = ref_gray.copy()
    flo_im_orig = flo_gray.copy()
    ref_gray = filters.normalize(ref_gray, 0.0, None)
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
    step_lengths = np.array([[1., 1.], [1., 0.5], [0.5, 0.1]])

    # Create the transform and add it to the registration framework (switch between affine/rigid transforms by commenting/uncommenting)
    # Affine
    reg.add_initial_transform(AffineTransform(2), np.array([1.0/diag, 1.0/diag, 1.0/diag, 1.0/diag, 1.0, 1.0]))
    # Rigid 2D
    #reg.add_initial_transform(Rigid2DTransform(2), np.array([1.0/diag, 1.0, 1.0]))

    # Set the parameters
    reg.set_iterations(param_iterations)
    reg.set_gradient_magnitude_threshold(0.001)
    reg.set_sampling_fraction(param_sampling_fraction)
    reg.set_step_lengths(step_lengths)
    reg.set_optimizer('sgd')

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

    ### Warp final image
    c = transforms.make_image_centered_transform(transform, ref_gray, flo_gray, spacing, spacing)

    # Print out transformation parameters
    print('Transformation parameters: %s.' % str(transform.get_params()))

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
    # Save the registered image   
    cv2.imwrite('./test_images/output/registered.tif', flo_im_warped)

    # Compute the absolute difference image between the reference and registered images
    D1 = np.abs(ref_im-flo_im_warped)
    err = np.mean(D1)
    print("Err: %f" % err)
    cv2.imwrite('./test_images/output/diff.png', D1)

    return True

if __name__ == '__main__':
    start_time = time.time()
    res = main()
    end_time = time.time()
    if res == True:
        print("Elapsed time: " + str((end_time-start_time)))
