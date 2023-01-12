import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.contrib import tzip

Mode = "CD31"       #  Image modality used for scaling 

def image_resize(im_dir, size):
    images = sorted(os.listdir(im_dir), key = lambda x: int(x[:-4]))
    for f in images:
        im = cv2.resize(cv2.imread(os.path.join(im_dir, f)), size)              # size = (w, h)
        cv2.imwrite(os.path.join(im_dir, f), im)


def im_show(img, w, h):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img", w, h)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def draw_contour(img, contours):
    new_img = np.zeros_like(img)
    drawed_img = cv2.drawContours(new_img, contours, -1, (255, 255, 255), 10)
    im_show(drawed_img, 600, 500)


def contourArea(img):
    '''
    Calculate contours Area of tissus example
    '''
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im_gray, 254, 255, cv2.THRESH_BINARY_INV) # cv.THRESH_BINARY+cv.THRESH_OTSU cv2.THRESH_BINARY_INV
    # im_show(thresh, 600, 500)
    if Mode == "H&E":
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 10)
    elif Mode == "CD31":
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 20)
    # im_show(thresh, 600, 500)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # draw_contour(thresh, contours)
    total_area = 0
    for contour in contours:
        total_area += cv2.contourArea(contour)
    return total_area


def sacle_images(im1_dir, im2_dir):
    '''
    scale registred images to original size

    im1_dir correspond to the registred images
    im2_dir correspond to the original images
    '''
    img_reg = sorted(os.listdir(im1_dir), key = lambda x: int(x[:-4]))
    img_orig = sorted(os.listdir(im2_dir), key = lambda x: int(x[:-4]))
    out_dir = "scaled_images"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for im1, im2 in tzip(img_reg, img_orig):
        img1 = cv2.imread(os.path.join(im1_dir, im1))
        img2 = cv2.imread(os.path.join(im2_dir, im2))
        area1 = contourArea(img1)                       # registred area
        area2 = contourArea(img2)                       # original area
        ratio = math.sqrt(area2 / area1)
        M = np.zeros((2, 3))
        M[0][0] = ratio
        M[1][1] = ratio
        M[0][2] = (img1.shape[1] * (1 - ratio)) / 2              # Need to move the image to Center, or it will be in left corner
        M[1][2] = (img1.shape[0] * (1 - ratio)) / 2  
        output = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), borderValue = (255, 255, 255))
        cv2.imwrite(os.path.join(out_dir, im1), output)


def getDiff(im_dir):
    imgs = sorted(os.listdir(im_dir), key = lambda x: int(x[:-4]))
    for im1, im2 in tzip(imgs, imgs[1:]):
        img1 = cv2.imread(os.path.join(im_dir, im1))
        img2 = cv2.resize(cv2.imread(os.path.join(im_dir, im2)), (img1.shape[1], img1.shape[0]))        
        D = np.abs(img1-img2)
        diff_dir = "diff"
        if not os.path.exists(diff_dir):
            os.makedirs(diff_dir)
        cv2.imwrite(os.path.join(diff_dir, 'diff_{}_{}.png'.format(im1.split('.')[0], im2.split('.')[0])), D)
    

def visual_for_ransac(num_points):
    np.random.seed(1)
    im1 = np.ones((1000, 1000, 3))
    im1.fill(255)
    points = np.random.randint(0, 1000, (num_points, 2), dtype = np.int64)
    for point in points:
        im1 = cv2.circle(im1, (point[0], point[1]), radius = 1, color = (0, 0, 255), thickness = 5)
    points_pertube = points + np.random.randint(0, 20, (num_points, 2), dtype = np.int64)
    points_pertube = np.clip(points_pertube, 0, 1000)
    for point in points_pertube:
        im1 = cv2.circle(im1, (point[0], point[1]), radius = 1, color = (0, 255, 0), thickness = 5)
    plt.imshow(im1[:, :, ::-1])
    plt.show()



if __name__ == '__main__':
    im1_dir = "images"
    im2_dir = "images_inpainted"
    im3_dir = "scaled_images"
    # M = cv2.getAffineTransform(p1, p2)
    # sacle_images(im2_dir, im3_dir)
    getDiff(im1_dir)

