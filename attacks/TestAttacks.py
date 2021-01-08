import cv2
import numpy as np
import pywt
import attacks.Attacks as attacks
import logging
import PIL
import DWT as dwt
logging.basicConfig(level=logging.DEBUG)

coverImagePath = 'images\\mandrill_512.jpg'
watermarkImagePath = 'images\\lenna_512.jpg'


dwt.DWT_RGB_LL(coverImagePath, watermarkImagePath, "rotate_image", 90)
dwt.DWT_RGB_LL(coverImagePath, watermarkImagePath, "distortion")
dwt.DWT_RGB_LL(coverImagePath, watermarkImagePath, "resize_attack", 200)

    # Image.open
dwt.DWT_RGB_LL(coverImagePath, watermarkImagePath, "compression", 1)
dwt.DWT_RGB_LL(coverImagePath, watermarkImagePath, "gaussian_noise")
dwt.DWT_RGB_LLoptions(coverImagePath, watermarkImagePath, "salt_and_pepper", 0.3, 100)

dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "rotate_image", 90)
dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "distortion")
dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "resize_attack", 200)
    # Image.open
dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "compression", 1)
dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "gaussian_noise")
dwt.DWT_GRAY_LL(coverImagePath, watermarkImagePath, "salt_and_pepper", 0.3, 100)

