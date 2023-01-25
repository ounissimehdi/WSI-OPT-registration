import scipy.ndimage.interpolation
from PIL import Image
import pandas as pd
from glob import glob
from natsort import natsorted
import numpy as np
import os
import argparse
import cv2

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-img_id" ,"--image_id", help="image to be registered",type=int)
    args = parser.parse_args()

    # Getting the acquisition ID
    image_id = args.image_id

    scale = 10
    # Pixel-size
    spacing = np.array([1.0, 1.0])

    dfs_list = []
    csv_paths = natsorted(glob(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), '*.csv')))
    # print(image_id+1)
    for csv_path in csv_paths: dfs_list.append(pd.read_csv(csv_path))

    os.makedirs(os.path.join('..', 'HE', 'estimated_para_images_10to100'), exist_ok = True)

    
    flo_im = Image.open(os.path.join('..', 'HE', 'clean_dataset', '0.tif'))

    # if scale_percent!=100:
    width = int(np.shape(flo_im)[1] * scale/ 100)
    height = int(np.shape(flo_im)[0] * scale/ 100)
    dim = (width, height)

    
    if len(glob(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), '0.tif'))) == 0:
        flo_im.resize(dim).save(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), '0.tif'))
    else:
        flo_im = Image.open(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), '0.tif'))

    flo_im = np.array(flo_im)

    #### GET RATIO
    flo_im_B = flo_im[..., 0]

    old_grid, switch_memory = 0, False
    for idx in range(image_id+1):
        
        df_tfs = dfs_list[idx]
        tfs_param = np.array([df_tfs['xx'][0], df_tfs['xy'][0], df_tfs['yx'][0], df_tfs['yy'][0],df_tfs['X'][0], df_tfs['Y'][0]])

        warp_grid, old_grid = compute_grid(tfs_param, np.zeros(flo_im_B.shape), spacing, old_grid, switch_memory)
        switch_memory = True
        # print(idx)
    
    files = natsorted(glob(os.path.join('..', 'HE', 'clean_dataset', '*.tif')))
    files = files[1:len(files)]

    # Transform the floating image into the reference image space by applying transformation 'c'
    flo_im   =  Image.open(files[image_id]).resize(dim).convert('L')
    flo_im   =  np.array(flo_im)

    rr,cc = np.where(flo_im<127)
    area_original = len(rr)

    # Create the output image
    flo_im_warped   = np.zeros(flo_im.shape)
    flo_im_warped.fill(255)

    grid_warp_img(warp_grid, flo_im, flo_im_warped, 255.0)

    rr,cc = np.where(flo_im_warped<127)
    area_registered = len(rr)

    ratio = np.sqrt(area_original/area_registered)

    ###### REGISTER THE ORIGINAL IMAGES

    flo_im = Image.open(files[image_id])
    flo_im   =  np.array(flo_im)

    #### GET RATIO
    flo_im_B = flo_im[..., 0]

    old_grid, switch_memory = 0, False
    for idx in range(image_id+1):
        
        df_tfs = dfs_list[idx]
        tfs_param = np.array([df_tfs['xx'][0], df_tfs['xy'][0], df_tfs['yx'][0], df_tfs['yy'][0],df_tfs['X'][0]*scale, df_tfs['Y'][0]*scale])

        warp_grid, old_grid = compute_grid(tfs_param, np.zeros(flo_im_B.shape), spacing, old_grid, switch_memory)
        switch_memory = True
        # print(idx)

    flo_im_B = flo_im[..., 0]
    flo_im_G = flo_im[..., 1]
    flo_im_R = flo_im[..., 2]

    # Create the output image
    flo_im_warped_B = np.zeros(flo_im_B.shape)
    flo_im_warped_G = np.zeros(flo_im_G.shape)
    flo_im_warped_R = np.zeros(flo_im_R.shape)
    flo_im_warped_B.fill(255)
    flo_im_warped_G.fill(255)
    flo_im_warped_R.fill(255)

    grid_warp_img(warp_grid, flo_im_B, flo_im_warped_B, 255.0)
    grid_warp_img(warp_grid, flo_im_G, flo_im_warped_G, 255.0)
    grid_warp_img(warp_grid, flo_im_R, flo_im_warped_R, 255.0)

    flo_im_warped = np.stack((flo_im_warped_B, flo_im_warped_G, flo_im_warped_R), axis = -1)
    # flo_im_warped = flo_im_warped.astype(np.uint8)


    M = np.zeros((2, 3))
    M[0][0] = ratio
    M[1][1] = ratio
    M[0][2] = (flo_im_warped.shape[1] * (1 - ratio)) / 2              # Need to move the image to Center, or it will be in left corner
    M[1][2] = (flo_im_warped.shape[0] * (1 - ratio)) / 2  
    output = cv2.warpAffine(flo_im_warped, M, (flo_im_warped.shape[1], flo_im_warped.shape[0]), borderValue = (255, 255, 255))

    flo_im_warped = output.astype(np.uint8)
    Image.fromarray(flo_im_warped).save(os.path.join('..', 'HE', 'estimated_para_images_10to100', os.path.basename(df_tfs['filename'][0])))
    Image.fromarray(flo_im_warped).resize(dim).save(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), os.path.basename(df_tfs['filename'][0])))