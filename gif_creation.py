from PIL import Image
from glob import glob
from natsort import natsorted
import numpy as np
import os

scale = 10
# Creating GIF animation for the whole sequence
files = natsorted(glob(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), '*.tif')))
images = []
for reg_path in files: images.append(Image.open(reg_path))
images[0].save(os.path.join('..', 'HE', 'PAR_images_scale_'+str(scale), 'registration_animation.gif'), save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)
