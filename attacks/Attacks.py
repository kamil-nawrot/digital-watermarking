import calendar
import time

import cv2
from PIL import Image


def checkPSNR(original, attacked):
    # img1 = cv2.imread("images/" + original)
    img1 = cv2.imread("../images/lenna_256.jpg")
    # img2 = cv2.imread("images/" + attacked)
    img2 = cv2.imread("compressed/Compressed_1606579782_lenna_256.jpg")
    psrn = cv2.PSNR(img1, img2)
    print(psrn)
    return psrn


def compression(filename, quality):
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    im = Image.open("images/" + filename)
    ts = calendar.timegm(time.gmtime())
    im.save("attacks/compressed/Compressed_" + str(quality) + "_" + str(ts) + "_" + filename, optimize=True, quality=quality)
    return ''


def distorition(filename):
    # https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
    # https://www.geeksforgeeks.org/python-distort-method-in-wand/
    return ''


def transformation(filename):
    # https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
    im = Image.open(filename)
    return ''


checkPSNR("", "")
