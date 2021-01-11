import logging

import cv2

import Attacks
import DWT_GRAY
import DWT_RGB
import DWT_SVD_GRAY
import DWT_SVD_RGB

logging.basicConfig(level=logging.DEBUG)

im_path = 'images/mandrill_512.jpg'
wm_path = 'images/lenna_512.jpg'

im = cv2.imread(im_path)
wm = cv2.imread(wm_path)


# use: im, im_wm, im_wm_atk, wm, wm_ext

def print_latex_format(psnrs_arr):
    print(" & ".join(psnrs_arr))


def convert_psnr_to_latex(psnr):
    return str(round(psnr, 3))


def base_psnr_for_attacked():
    print("SP & GN & cmprs & rotate")
    psnrs = []
    # psnrs.append(convert_psnr_to_latex(Attacks.check_psnr(wm, wm_path)))
    psnrs.append(convert_psnr_to_latex(cv2.PSNR(wm, Attacks.salt_pepper(wm_path))))
    psnrs.append(convert_psnr_to_latex(cv2.PSNR(wm, Attacks.gaussian_noise(wm_path))))
    psnrs.append(convert_psnr_to_latex(cv2.PSNR(wm, Attacks.compression(wm_path, 20))))
    psnrs.append(convert_psnr_to_latex(cv2.PSNR(wm, Attacks.rotate_image(wm_path))))
    print_latex_format(psnrs)


# ________________ DWT
# ______________________________________________________
def test_DWT_GRAY_LL():
    print("dwt gray LL")
    image_with_watermark = DWT_GRAY.DWT_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        if len(atck_im.shape) > 2:
            atck_im = cv2.cvtColor(atck_im, cv2.COLOR_BGR2GRAY)
        wm_ext = DWT_GRAY.DWT_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_GRAY_HL():
    print("dwt gray HL")
    image_with_watermark = DWT_GRAY.DWT_GRAY_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        if len(atck_im.shape) > 2:
            atck_im = cv2.cvtColor(atck_im, cv2.COLOR_BGR2GRAY)
        wm_ext = DWT_GRAY.DWT_GRAY_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_RGB_LL():
    print("dwt RGB LL")
    image_with_watermark = DWT_RGB.DWT_RGB_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_RGB.DWT_RGB_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ________________ DWT-SVD
# ______________________________________________________
def test_DWT_RGB_HL():
    print("dwt RGB HL")
    image_with_watermark = DWT_RGB.DWT_RGB_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_RGB.DWT_RGB_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_SVD_GRAY_LL():
    print("dwt svd gray LL")
    image_with_watermark = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        if len(atck_im.shape) > 2:
            atck_im = cv2.cvtColor(atck_im, cv2.COLOR_BGR2GRAY)
        wm_ext = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_SVD_GRAY_HL():
    print("dwt svd gray HL")
    image_with_watermark = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        if len(atck_im.shape) > 2:
            atck_im = cv2.cvtColor(atck_im, cv2.COLOR_BGR2GRAY)
        wm_ext = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_SVD_RGB_LL():
    print("dwt svd rgb LL")
    image_with_watermark = DWT_SVD_RGB.DWT_SVD_RGB_LL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD_RGB.DWT_SVD_RGB_LL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_DWT_SVD_RGB_HL():
    print("dwt svd rgb HL")
    image_with_watermark = DWT_SVD_RGB.DWT_SVD_RGB_HL_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = DWT_SVD_RGB.DWT_SVD_RGB_HL_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ________________ DWT-DCT

base_psnr_for_attacked()

# ______ DWT
# test_DWT_GRAY_LL()
# test_DWT_GRAY_HL()
# test_DWT_RGB_LL()
# test_DWT_RGB_HL()

# ______ DWT-SVD
# test_DWT_SVD_GRAY_LL()
# test_DWT_SVD_GRAY_HL()
# test_DWT_SVD_RGB_LL()
# test_DWT_SVD_RGB_HL()
