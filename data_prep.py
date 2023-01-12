from PIL import Image
import numpy as np
from natsort import natsorted
from glob import glob
import os
from tqdm import tqdm

path_dataset = natsorted(glob(os.path.join('..', 'cd31_sent_to_Haocheng', '*.tif')))

## Getting the smallest images size

# Init Smallest shape width and hight
w_sz, h_sz = 100000, 100000
for path in tqdm(path_dataset):
    tmp_h, tmp_w, _ = np.shape(Image.open(path))
    if tmp_w<w_sz: w_sz = tmp_w
    if tmp_h<h_sz: h_sz = tmp_h

print('Smallest shape', w_sz, h_sz)

# h_sz, w_sz = 4993, 5980

# Setting the points for cropped image
left = 0
top = 0
right = w_sz
bottom = 4993


os.makedirs(os.path.join('..', 'clean_dataset'), exist_ok = True)
for path in tqdm(path_dataset):
    org_img = Image.open(path).crop((left, top, right, bottom)).save(os.path.join('..', 'clean_dataset', os.path.basename(path)))

