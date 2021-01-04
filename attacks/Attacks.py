import calendar
import time

import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise

IMAGES_DIR = "../images/"


def checkPSNR(original, attacked):
    img1 = cv2.imread(IMAGES_DIR + original)
    print(img1)
    print('0000000000000000000000000000')
    print(attacked)
    # img2 = cv2.imread("compressed/Compressed_1606579782_lenna_256.jpg")
    psrn = cv2.PSNR(img1, attacked)

    print(psrn)
    return psrn


def compression(filename, quality):
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    im = Image.open(IMAGES_DIR + filename)
    ts = calendar.timegm(time.gmtime())
    im.save("../attacks/compressed/Compressed_" + str(quality) + "_" + str(ts) + "_" + filename, optimize=True, quality=quality)
    return ''


def distorition(filename):
    # https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
    # https://www.geeksforgeeks.org/python-distort-method-in-wand/
    return ''


def resize_attack(filename, scale_percent):
    img = cv2.imread(IMAGES_DIR + filename)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # print('Resized Dimensions : ', resized.shape)
    # cv2.imshow("Resized image", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return resized


def gaussian_noise(filename):
    im = Image.open(IMAGES_DIR + filename)
    im_arr = np.asarray(im)
    # can parametrize: clip, mean, var
    noise_img = random_noise(im_arr, mode='gaussian')
    noise_img = (255 * noise_img).astype(np.uint8)
    img = Image.fromarray(noise_img)
    return noise_img


def salt_and_pepper(filename):
    im = Image.open(IMAGES_DIR + filename)
    im_arr = np.asarray(im)
    # can parametrize amount <0, 1> and salt_vs_pepper  <0, 1>
    noise_img = random_noise(im_arr, mode='s&p', salt_vs_pepper=0.5)
    noise_img = (255 * noise_img).astype(np.uint8)
    Image.fromarray(noise_img).show()
    return noise_img


# compression("lenna_512.jpg", 50)
# salt_and_pepper("lenna_256.jpg")
# gaussian_noise("lenna_256.jpg")
# checkPSNR("lenna_256.jpg", gaussian_noise("lenna_256.jpg"))
resize_attack("lenna_256.jpg", 200)
resize_attack("lenna_256.jpg", 50)


# KAMIL PSEUDO CODE/ SOME KIND OF CODE


def performAttacks():
    return


# arrayWithExtractedWatermarks
# Attacks for DWT_GRAY_LL
# watermarkedImage = DWT.DWT_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
# watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
# DWT_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

# Attack for DWT_SVD_GRAY_LL
# watermarkedImage = DWT.DWT_SVD_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
# watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
# DWT_SVD_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

def performAttacksOnWatermarkedImage(imageWithWatermark, method):  # method = DWT.DWT_GRAY_LL
    return


# saltAndPepperAttackResult = saltAndPepper(imageWithWatermark)
# cv2.imwrite(''+method+'_watermarkedImageAfterSaltAndPepperAttack', saltAndPepperAttackResult) #zapis
# GausianAttackResult = gausian(imageWithWatermark)
# cv2.imwrite
# transformationAttackResult = transformation(imageWithWatermark)
# cv2.imwrite
# compressionAttackResult = compression(imageWithWatermark)
# cv2.imwrite

# attacksTable = [pathTosaltAndPepperAttack,pathToGausianAttack, pathToTransformationAttack, pathToCompressionAttack ]

# return attacksTable

def extractWatermarksFromArray(tableWithAttackedImagesPaths, method):  # method = DWT.DWT_GRAY_LL
    return


# counter = 0
# method + "_EXTRACT"
# extractedWatermarksArray

# for path in tableWithAttackedImagesPaths
# extractedWatermark = runAppropriateExtracMethod(path, method)
# extractedWatermarksArray[counter] = extractedWatermark
# counter ++

# return extractedWatermarksArray


def runAppropriateExtracMethod(pathToWatermarkedImageAfterAttack, method):  # method = DWT.DWT_GRAY_LL
    return


# coverImagePath= "..."
# watermarkImagePath = "..."
# extractedWatermark
# switch(Method)
# case DWT.DWT_GRAY_LL
#   extractedWatermark= DWT.DWT_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, pathToWatermarkedImageAfterAttack )
# break
# case DWT.DWT_SVD_GRAY_LL
#   extractedWatermark = DWT.DWT_SVD_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, pathToWatermarkedImageAfterAttack ) #added _ExTRACT TO METHOD NAME
# break
# default ...

# return extractedWatermark

def getExtractedWatermarkNormalWay(coverImagePath, watermarkImagePath, method):
    return


# extractedWatermark = runAppropriateImageWatermarkingMethod(coverImagePath,watermarkImagePath, method)
# return extracted watermark

def runAppropriateImageWatermarkingMethod(method):  # method = DWT.DWT_GRAY_LL
    return


# extractedWatermark
# switch(Method)
# case DWT.DWT_GRAY_LL
#   watermarkedImagePath= DWT.DWT_GRAY_LL(coverImagePath,watermarkImagePath) #no added _EXTRACT TO METHOD NAME
#   extractedWatermark = DWT.DWT_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, watermarkedImagePath)
# break
# case DWT.DWT_SVD_GRAY_LL
#   watermarkedImagePath= DWT_SVD.DWT_GRAY_LL(coverImagePath,watermarkImagePath) #no added _EXTRACT TO METHOD NAME
#   extractedWatermark = DWT.DWT_SVD_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, watermarkedImagePath)
# break
# default ...

# return extractedWatermark

def compareWatermarks(watermarkAfterAttack, watermarkAfterNormalExtraction):
    return
# psnr(watermarkAfterAttack)
# psnr(watermarkAfterNormalExtraction)
# DO SMTH
