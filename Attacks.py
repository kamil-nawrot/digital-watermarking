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
def rotate_image(im_path):
    # print("rotation")
    im = cv2.imread(im_path)
    rotated = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    save_attacked_image(rotated, "rot")
    return rotated


# cv2.imread(...)
def distorition(im_path):
    img = cv2.imread(im_path)
    color_img = img
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    A = img.shape[0] / 3.0
    w = 2.0 / img.shape[1]
    shift = lambda x: A * np.sin(1 * np.pi * x * w)
    for i in range(img.shape[0]):
        img[:, i] = np.roll(img[:, i], int(shift(i)))
    return img


# cv.imread(...)
def resize_attack(im_path, scale_percent):
    im = cv2.imread(im_path)
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)


# Image.open(...)
def compression(im_path, quality):
    # print("compression")
    img = Image.open(im_path)
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    ts = calendar.timegm(time.gmtime())
    path = "attacks/cmprs_" + str(quality) + "_" + str(ts) + ".jpg"
    img.save(path, optimize=True, quality=quality)
    return cv2.imread(path)


# Image.open(...)
def gaussian_noise(im_path):
    # print("gaussian")
    img = Image.open(im_path)
    im_arr = np.asarray(img)
    # can parametrize: clip, mean, var
    noise_img = random_noise(im_arr, mode='gaussian')
    noise_img = (255 * noise_img).astype(np.uint8)
    save_attacked_image(noise_img, "gn")
    return noise_img


# Image.open(...)
def salt_pepper(im_path):
    # print("salt pepper")
    img = Image.open(im_path)
    im_arr = np.asarray(img)
    # can parametrize amount <0, 1> and salt_vs_pepper  <0, 1>
    noise_img = random_noise(im_arr, mode='s&p', amount=0.3, salt_vs_pepper=0.5)
    noise_img = (255 * noise_img).astype(np.uint8)
    save_attacked_image(noise_img, "sp")
    return noise_img


def perform_all_attacks_on_watermarked_image(im_wm_path):
    result_salt_pepper = salt_pepper(im_wm_path)

    result_gaussian = gaussian_noise(im_wm_path)

    result_compression = compression(im_wm_path, 20)

    result_rotate = rotate_image(im_wm_path)

    # result_resize = resize_attack(im_wm_path, 200)
    # result_distortion = distorition(im_wm_path)
    return [result_salt_pepper, result_gaussian, result_compression, result_rotate]


def save_attacked_image(im, attack_type):
    ts = calendar.timegm(time.gmtime())
    cv2.imwrite("attacks/" + attack_type + "_" + str(ts) + ".jpg", im)
