import calendar
import time

import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise


# cv2.imread(...)
def check_psnr(wm, wm_ext_path):
    wm_ext = cv2.imread(wm_ext_path)
    psnr = cv2.PSNR(wm, wm_ext)
    return psnr


# cv2.imread(...)
def rotate_image(im, angle):
    print('attack rotate')
    row, col, colors = im.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(im, rot_mat, (col, row))


# cv2.imread(...)
def distorition(im):
    print('attack distorition')
    color_img = im
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    A = img.shape[0] / 3.0
    w = 2.0 / img.shape[1]
    shift = lambda x: A * np.sin(1 * np.pi * x * w)
    for i in range(img.shape[0]):
        img[:, i] = np.roll(img[:, i], int(shift(i)))
    return img


# cv.imread(...)
def resize_attack(im, scale_percent):
    print('attack resize')
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)


# Image.open(...)
def compression(im_path, quality):
    print('attack compression')
    img = Image.open(im_path)
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    ts = calendar.timegm(time.gmtime())
    path = "attacks/compressed/Compressed_" + str(quality) + "_" + str(ts) + ".jpg"
    img.save(path, optimize=True, quality=quality)
    return cv2.imread(path)


# Image.open(...)
def gaussian_noise(im_path):
    print('attack gaussian')
    img = Image.open(im_path)
    im_arr = np.asarray(img)
    # can parametrize: clip, mean, var
    noise_img = random_noise(im_arr, mode='gaussian')
    noise_img = (255 * noise_img).astype(np.uint8)
    return noise_img


# Image.open(...)
def salt_pepper(im_path):
    print('attack salt pepper')
    img = Image.open(im_path)
    im_arr = np.asarray(img)
    # can parametrize amount <0, 1> and salt_vs_pepper  <0, 1>
    noise_img = random_noise(im_arr, mode='s&p', amount=0.5, salt_vs_pepper=0.5)
    noise_img = (255 * noise_img).astype(np.uint8)
    return noise_img


def perform_all_attacks_on_watermarked_image(im_wm_path):
    result_salt_pepper = salt_pepper(im_wm_path)
    result_gaussian = gaussian_noise(im_wm_path)
    result_compression = compression(im_wm_path, 25)
    # result_resize = resize_attack(im_wm, 2)
    # result_rotate = rotate_image(im_wm, 90)
    # result_distortion = distorition(im_wm)
    return [result_salt_pepper, result_gaussian, result_compression]
