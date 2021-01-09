import logging

import cv2

import Attacks
import DWT_GRAY
import DWT_SVD_GRAY
import DWT_SVD_RGB
from DWT_DCT import DWTDCT

logging.basicConfig(level=logging.DEBUG)

im_path = 'images/mandrill_512.jpg'
wm_path = 'images/lenna_512.jpg'

im = cv2.imread(im_path)
wm = cv2.imread(wm_path)


# use: im, im_wm, im_wm_atk, wm, wm_ext

# ______________________________________________________
def test_DWT_SVD_RGB_LL():
    image_with_watermark = DWT_SVD_RGB.DWT_SVD_RGB_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for im_wm_atk in attacked_images:
        wm_ext = DWT_SVD_RGB.DWT_SVD_RGB_LL_EXTRACT(im_path, wm_path, im_wm_atk)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(str(round(psnr, 3)))
    print(" & ".join(psnrs))


# ______________________________________________________
def test_DWT_SVD_RGB_HL():
    image_with_watermark = DWT_SVD_RGB.DWT_SVD_RGB_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD_RGB.DWT_SVD_RGB_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(str(round(psnr, 3)))
    print(" & ".join(psnrs))


# ______________________________________________________
def test_DWT_SVD_GRAY_LL():
    image_with_watermark = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(str(round(psnr, 3)))
    print(" & ".join(psnrs))


# ______________________________________________________
def test_DWT_SVD_GRAY_HL():
    image_with_watermark = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(str(round(psnr, 3)))
    print(" & ".join(psnrs))


# ______________________________________________________
def test_DWT_GRAY_LL():
    image_with_watermark = DWT_GRAY.DWT_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_GRAY.DWT_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(str(round(psnr, 3)))
    print(" & ".join(psnrs))


def test_DWT_DCT():
    baseImage = DWTDCT("base", "images/lenna_256.jpg", (1024, 1024))
    originalImage = DWTDCT("base", "images/lenna_256.jpg", (1024, 1024))
    watermarkImage = DWTDCT("watermark", "images/mandrill_256.jpg", (128, 128))

    baseImage.embed_watermark('HL', watermarkImage)
    baseImage.display()
    baseImage.display_difference(originalImage)
    baseImage.save('processed_images/watermarked_image.jpg')

    reconstructedImage = DWTDCT("watermarked", "watermarked_image.jpg")
    reconstructedImage.extract_watermark('HL', 128)


# test_DWT_GRAY_LL()
# test_DWT_RGB_LL()

test_DWT_SVD_GRAY_LL()
test_DWT_SVD_GRAY_HL()

test_DWT_SVD_RGB_LL()
test_DWT_SVD_RGB_HL()
