import scipy.ndimage.interpolation
from PIL import Image
import pandas as pd
from glob import glob
from natsort import natsorted
import numpy as np
import os

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
    scale = 10
    # Pixel-size
    spacing = np.array([1.0, 1.0])

    dfs_list = []
    csv_paths = natsorted(glob(os.path.join('..', 'PAR_images_scale_'+str(scale), '*.csv')))
    for csv_path in csv_paths: dfs_list.append(pd.read_csv(csv_path))

    os.makedirs(os.path.join('..','estimated_para_images_10to100'), exist_ok = True)

    flo_im = Image.open(os.path.join('..', 'clean_dataset', '0.tif'))
    
    # if scale_percent!=100:
    width = int(np.shape(flo_im)[1] * scale/ 100)
    height = int(np.shape(flo_im)[0] * scale/ 100)
    dim = (width, height)

    flo_im.resize(dim).save(os.path.join('..', 'PAR_images_scale_'+str(scale), '0.tif'))


    old_grid, switch_memory = 0, False
    for i in range(len(dfs_list)):

        df_tfs = dfs_list[i]
        flo_im = Image.open(os.path.join('..', 'clean_dataset',os.path.basename(df_tfs['filename'][0])))
        
        # # resize image
        # flo_im = flo_im.resize(dim)
        
        # tfs_param = []
        # for tf in range(i+1):
        tfs_param = np.array([df_tfs['xx'][0], df_tfs['xy'][0], df_tfs['yx'][0], df_tfs['yy'][0],df_tfs['X'][0]*scale, df_tfs['Y'][0]*scale])
        
        # print(tfs_param)
        # Transform the floating image into the reference image space by applying transformation 'c'
        flo_im   =  np.array(flo_im)

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

        warp_grid, old_grid = compute_grid(tfs_param, flo_im_warped_B, spacing, old_grid, switch_memory)
        switch_memory = True

        grid_warp_img(warp_grid, flo_im_B, flo_im_warped_B, 255.0)
        grid_warp_img(warp_grid, flo_im_G, flo_im_warped_G, 255.0)
        grid_warp_img(warp_grid, flo_im_R, flo_im_warped_R, 255.0)

        flo_im_warped = np.stack((flo_im_warped_B, flo_im_warped_G, flo_im_warped_R), axis = -1)
        flo_im_warped = flo_im_warped.astype(np.uint8)

        Image.fromarray(flo_im_warped).save(os.path.join('..','estimated_para_images_10to100', os.path.basename(df_tfs['filename'][0])))
        Image.fromarray(flo_im_warped).resize(dim).save(os.path.join('..','PAR_images_scale_'+str(scale), os.path.basename(df_tfs['filename'][0])))

    # Creating GIF animation for the whole sequence
    files = natsorted(glob(os.path.join('..', 'PAR_images_scale_'+str(scale), '*.tif')))
    images = []
    for reg_path in files: images.append(Image.open(reg_path))
    images[0].save(os.path.join('..', 'PAR_images_scale_'+str(scale), 'registration_animation.gif'), save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)
