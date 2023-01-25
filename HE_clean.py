from glob import glob
from natsort import natsorted
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

reg_HE_dataset = natsorted(glob(os.path.join('..', 'HE','HE_reg_dataset', '*.tif')))
HE_dataset = natsorted(glob(os.path.join('..', 'HE', 'HE_dataset', '*.tif')))

clean_HE_dataset_out = os.path.join('..', 'HE', 'clean_dataset')
os.makedirs(clean_HE_dataset_out, exist_ok = True)

w_sz, h_sz = 100000, 100000

for HE_path in tqdm(HE_dataset):
    for HE_reg_path in reg_HE_dataset:
        if os.path.basename(HE_path) == os.path.basename(HE_reg_path):
            shutil.copyfile(HE_path, os.path.join(clean_HE_dataset_out, os.path.basename(HE_path)))

            tmp_h, tmp_w, _ = np.shape(Image.open(HE_path))
            if tmp_w<w_sz: w_sz = tmp_w
            if tmp_h<h_sz: h_sz = tmp_h

print('Smallest shape', w_sz, h_sz)

# h_sz, w_sz = 4993, 5980
# 5924 4986
# Setting the points for cropped image
left = 0
top = 0
right = 5924
bottom = 4986

path_dataset = natsorted(glob(os.path.join('..', 'HE', 'clean_dataset', '*.tif')))
for path in tqdm(path_dataset):
    org_img = Image.open(path).crop((left, top, right, bottom)).save(os.path.join('..', 'HE', 'clean_dataset', os.path.basename(path)))

