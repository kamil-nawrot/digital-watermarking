import calendar
import logging
import time

import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise

import DWT
import DWT_SVD

logging.basicConfig(level=logging.DEBUG)


# cv2.imread(...)
def check_psnr(original, attacked):
    print(cv2.PSNR(original, attacked))
    return cv2.PSNR(original, attacked)


# cv2.imread(...)
def rotate_image(im, angle):
    row, col, colors = im.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(im, rot_mat, (col, row))


# cv2.imread(...)
def distorition(im):
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
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)


######################## works only with Image.open()

# Image.open(...)
def compression(im, quality):
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    ts = calendar.timegm(time.gmtime())
    path = "../attacks/compressed/Compressed_" + str(quality) + "_" + str(ts) + ".jpg"
    im.save(path, optimize=True, quality=quality)
    return cv2.imread(path)


# Image.open(...)
def gaussian_noise(im):
    im_arr = np.asarray(im)
    # can parametrize: clip, mean, var
    noise_img = random_noise(im_arr, mode='gaussian')
    noise_img = (255 * noise_img).astype(np.uint8)
    img = Image.fromarray(noise_img)  # remove ONLY if it works with Kamil's idea
    return noise_img


# Image.open(...)
def salt_pepper(im, amount, ratio):
    im_arr = np.asarray(im)
    # can parametrize amount <0, 1> and salt_vs_pepper  <0, 1>
    noise_img = random_noise(im_arr, mode='s&p', amount=amount, salt_vs_pepper=ratio)
    noise_img = (255 * noise_img).astype(np.uint8)
    # Image.fromarray(noise_img).show()
    return noise_img





# KAMIL PSEUDO CODE/ SOME KIND OF CODE

IMAGES_DIR = "processed_images\\"


def perform_attack(watermarked_img, invoked_by, attack_method, *args):
    # switcher = {
    #     "rotate_image": rotate_image(watermarked_img, args[0]),
    #     "distortion": distorition(watermarked_img),
    #     "resize_attack": resize_attack(watermarked_img, args[0]),
    #     "compression": compression(watermarked_img, args[0]),
    #     "gaussian_noise": gaussian_noise(watermarked_img),
    #     "salt_and_pepper": salt_and_pepper(watermarked_img, args[0],args[1]),
    # }
    # imageAfterAttack = switcher[attack_method]
    image_after_attack = None
    if attack_method == "rotate_image":
        image_after_attack = rotate_image(watermarked_img, args[0])
    elif attack_method == "distortion":
        image_after_attack = distorition(watermarked_img)
    elif attack_method == "resize_attack":
        image_after_attack = resize_attack(watermarked_img, args[0])
    elif attack_method == "compression":
        image_after_attack = compression(watermarked_img, args[0])
    elif attack_method == "gaussian_noise":
        image_after_attack = gaussian_noise(watermarked_img)
    elif attack_method == "salt_pepper":
        image_after_attack = salt_pepper(watermarked_img, args[0], args[1])
    else:
        raise Exception('Wrong attack name')

    logging.debug('before' + IMAGES_DIR + 'attacked_watermarked_img_' + invoked_by + '_' + attack_method + '.jpg')
    cv2.imwrite('' + IMAGES_DIR + 'attacked_watermarked_img_' + invoked_by + '_' + attack_method + '.jpg', image_after_attack.astype(np.uint8))
    logging.debug('after' + IMAGES_DIR + 'attacked_watermarked_img_' + invoked_by + '_' + attack_method + '.jpg')
    return image_after_attack


# arrayWithExtractedWatermarks
# Attacks for DWT_GRAY_LL
# watermarkedImage = DWT.DWT_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
# watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
# DWT_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

# Attack for DWT_SVD_GRAY_LL
# watermarkedImage = DWT.DWT_SVD_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
# watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
# DWT_SVD_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

def perform_all_attacks_on_watermarked_image(im_wm, method):  # method = DWT.DWT_GRAY_LL
    result_salt_pepper = salt_pepper(im_wm, 0.1, 0.5)
    result_gaussian = gaussian_noise(im_wm)
    result_resize = resize_attack(im_wm, 2)
    result_compression = compression(im_wm, 25)
    result_rotate = rotate_image(im_wm, 90)
    result_distortion = distorition(im_wm)

    return [result_salt_pepper, result_gaussian, result_resize, result_compression, result_rotate, result_distortion]


def extract_watermarks_from_images(images, method):  # method = DWT.DWT_GRAY_LL
    method + "_EXTRACT"  # wtf ???
    extracted_watermarks = []

    for path in images:
        extracted_wm = run_appropriate_extract_method(path, method)
        extracted_watermarks.append(extracted_wm)

    return extracted_watermarks


def run_appropriate_extract_method(im_with_wm_after_attack, method):  # method = DWT.DWT_GRAY_LL
    im_path = "..."
    wm_path = "..."

    methods = {
        "DWT.DWT_GRAY_LL": "DWT.DWT_GRAY_LL_EXTRACT",
        "DWT.DWT_SVD_GRAY_LL": "DWT.DWT_SVD_GRAY_LL_EXTRACT"
    }
    chosen_method = methods.get(method, "wrong method")
    return chosen_method(im_path, wm_path, im_with_wm_after_attack)


def get_extracted_watermark_normal_way(im_path, wm_path, method):
    return run_appropriate_image_watermarking_method(im_path, wm_path, method)


def run_appropriate_image_watermarking_method(im_path, wm_path, method):  # method = DWT.DWT_GRAY_LL

    if method == "DWT.DWT_GRAY_LL":
        im_with_wm_path = DWT.DWT_GRAY_LL(im_path, wm_path)
        return DWT.DWT_GRAY_LL_EXTRACT(im_path, wm_path, im_with_wm_path)
    elif method == "DWT.DWT_SVD_GRAY_LL":
        im_with_wm_path = DWT_SVD.DWT_GRAY_LL(im_path, wm_path)
        return DWT.DWT_SVD_GRAY_LL_EXTRACT(im_path, wm_path, im_with_wm_path)

    return


def compare_watermarks(watermarkAfterAttack, watermarkAfterNormalExtraction):
    return
# psnr(watermarkAfterAttack)
# psnr(watermarkAfterNormalExtraction)
# DO SMTH


# _____________________________ PLAYGROUND
# small_lena = "lenna_256.jpg"
# big_lena = "lenna_512.jpg"
# input_image = Image.open("../images/" + big_lena)
# input_image_cv = cv2.imread("../images/" + big_lena)
#
# compression(input_image, 1)
#
# salt_and_pepper(input_image, 0.1, 1)
# # gaussian_noise("lenna_256.jpg")
# # checkPSNR("lenna_256.jpg", gaussian_noise("lenna_256.jpg"))
# # resize_attack("lenna_256.jpg", 200)
# # resize_attack("lenna_256.jpg", 50)
# rotate_image(input_image_cv, 90)
# # distorition(big_lena)
# check_psnr(input_image_cv, salt_and_pepper(input_image, 0.1, 1))
# check_psnr(input_image_cv, input_image_cv)
