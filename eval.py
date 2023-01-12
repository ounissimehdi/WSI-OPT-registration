import os
import cv2
import torch
from piq import ssim, psnr, multi_scale_ssim, fsim, mdsi, gmsd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib import tzip

def getResult(im_dir, metric, mean = None):
    images = sorted(os.listdir(im_dir), key = lambda x: int(x[:-4]))
    result = []
    for f1, f2 in tzip(images, images[1:]):
        ref_im = cv2.imread(os.path.join(im_dir, f1))
        ref_im = torch.from_numpy(ref_im).permute(2, 0, 1)[None, ...]
        float_im = cv2.imread(os.path.join(im_dir, f2))
        float_im = torch.from_numpy(float_im).permute(2, 0, 1)[None, ...]
        
        m = metric(float_im, ref_im, data_range = 255)            # Creat SSIM metric
        result.append(m.item())
        if mean is not None:
            mean += m
    mean /= len(result) if mean is not None else None
    return result, mean



if __name__ == "__main__":
    im_dir = "images_inpainted"
    num_images = len(os.listdir(im_dir))
    x = np.arange(num_images)

    metrics = [ssim, psnr, multi_scale_ssim, fsim, mdsi, gmsd]
    dataframe = pd.DataFrame()
    for metric in metrics:
        result, mean = getResult(im_dir, metric, 0)
        dataframe[metric.__name__] = result
        dataframe[metric.__name__ + "_mean"] = mean.item()
    dataframe.to_csv("metric_original.csv", index=False, sep=',')
    
    




    

