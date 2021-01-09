import logging

import cv2

import Attacks
import DWT
import DWT_SVD

logging.basicConfig(level=logging.DEBUG)

im_path = 'images/mandrill_512.jpg'
wm_path = 'images/lenna_512.jpg'

im = cv2.imread(im_path)
wm = cv2.imread(wm_path)


# use: im, im_wm, im_wm_atk, wm, wm_ext

# ______________________________________________________
def test_DWT_SVD_RGB_LL():
    image_with_watermark = DWT_SVD.DWT_SVD_RGB_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for im_wm_atk in attacked_images:
        wm_ext = DWT_SVD.DWT_SVD_RGB_LL_EXTRACT(im_path, wm_path, im_wm_atk)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(psnr)
    print(psnrs)


# ______________________________________________________
def test_DWT_SVD_RGB_HL():
    image_with_watermark = DWT_SVD.DWT_SVD_RGB_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD.DWT_SVD_RGB_HL_EXTRACT(im_path, wm_path, atck_im)
        psnrs.append(Attacks.check_psnr(wm, wm_ext))
    print(psnrs)


# ______________________________________________________
def test_DWT_SVD_GRAY_LL():
    image_with_watermark = DWT_SVD.DWT_SVD_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD.DWT_SVD_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnrs.append(Attacks.check_psnr(wm, wm_ext))
    print(psnrs)


# ______________________________________________________
def test_DWT_SVD_GRAY_HL():
    image_with_watermark = DWT_SVD.DWT_SVD_GRAY_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD.DWT_SVD_GRAY_HL_EXTRACT(im_path, wm_path, atck_im)
        psnrs.append(Attacks.check_psnr(wm, wm_ext))
    print(psnrs)


# ______________________________________________________
def test_DWT_GRAY_LL():
    image_with_watermark = DWT.DWT_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT.DWT_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnrs.append(Attacks.check_psnr(wm, wm_ext))
    print(psnrs)


test_DWT_SVD_RGB_LL()
test_DWT_SVD_RGB_HL()
# test_DWT_SVD_GRAY_LL()
# test_DWT_SVD_GRAY_HL()
# test_DWT_GRAY_LL()
