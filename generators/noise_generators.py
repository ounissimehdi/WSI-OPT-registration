# Import Numpy/Scipy
import numpy as np
import scipy as sp

def _setup_additive_gaussian_noise(sigma):
    def apply(image):
        return image + np.random.normal(0.0, sigma, image.shape)
    return apply

def _apply_noise(image, func, clip=False):
    noisy_image = func(image)
    if clip == True:
        return np.clip(noisy_image, 0.0, 1.0)
    else:
        return noisy_image

def additive_gaussian_noise(image, sigma, clip=False):
    func = _setup_additive_gaussian_noise(sigma)
    return _apply_noise(image, func, clip)
