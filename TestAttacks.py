import logging

import cv2

import Attacks
import DWT_GRAY
import DWT_RGB
import DWT_SVD_GRAY
import DWT_SVD_RGB
import SVD_GRAY
import SVD_RGB

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


# ________________ SVD
# ______________________________________________________
def test_SVD_GRAY():
    print("SVD GRAY")
    image_with_watermark = SVD_GRAY.SVD_GRAY_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        if len(atck_im.shape) > 2:
            atck_im = cv2.cvtColor(atck_im, cv2.COLOR_BGR2GRAY)
        wm_ext = SVD_GRAY.SVD_GRAY_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ______________________________________________________
def test_SVD_RGB():
    print("SVD RGB")
    image_with_watermark = SVD_RGB.SVD_RGB_EMBED(im_path, wm_path)

    attacked_images = Attacks.perform_all_attacks_on_watermarked_image(image_with_watermark)

    psnrs = []
    for atck_im in attacked_images:
        wm_ext = SVD_RGB.SVD_RGB_EXTRACT(im_path, wm_path, atck_im)
        psnr = Attacks.check_psnr(wm, wm_ext)
        psnrs.append(convert_psnr_to_latex(psnr))
    print_latex_format(psnrs)


# ________________ DWT-SVD
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


# ________________ not attacked
# ______________________________________________________
def test_GRAY():
    print("test GRAY")
    psnrs = []
    # dwt LL
    im_wm1 = DWT_GRAY.DWT_GRAY_LL_EMBED(im_path, wm_path)
    gray_wm = cv2.cvtColor(cv2.imread(im_wm1), cv2.COLOR_BGR2GRAY)
    ext1 = DWT_GRAY.DWT_GRAY_LL_EXTRACT(im_path, wm_path, gray_wm)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt HL
    im_wm1 = DWT_GRAY.DWT_GRAY_HL_EMBED(im_path, wm_path)
    gray_wm = cv2.cvtColor(cv2.imread(im_wm1), cv2.COLOR_BGR2GRAY)
    ext1 = DWT_GRAY.DWT_GRAY_HL_EXTRACT(im_path, wm_path, gray_wm)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # svd
    im_wm1 = SVD_GRAY.SVD_GRAY_EMBED(im_path, wm_path)
    gray_wm = cv2.cvtColor(cv2.imread(im_wm1), cv2.COLOR_BGR2GRAY)
    ext1 = SVD_GRAY.SVD_GRAY_EXTRACT(im_path, wm_path, gray_wm)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-svd LL
    im_wm1 = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EMBED(im_path, wm_path)
    gray_wm = cv2.cvtColor(cv2.imread(im_wm1), cv2.COLOR_BGR2GRAY)
    ext1 = DWT_SVD_GRAY.DWT_SVD_GRAY_LL_EXTRACT(im_path, wm_path, gray_wm)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-svd HL
    im_wm1 = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EMBED(im_path, wm_path)
    gray_wm = cv2.cvtColor(cv2.imread(im_wm1), cv2.COLOR_BGR2GRAY)
    ext1 = DWT_SVD_GRAY.DWT_SVD_GRAY_HL_EXTRACT(im_path, wm_path, gray_wm)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-dct LL
    # dwt-dct HL
    print_latex_format(psnrs)


def test_RGB():
    print("test RGB")
    psnrs = []
    # dwt LL
    im_wm1 = DWT_RGB.DWT_RGB_LL_EMBED(im_path, wm_path)
    im_wm1 = cv2.imread(im_wm1, 8)
    ext1 = DWT_RGB.DWT_RGB_LL_EXTRACT(im_path, wm_path, im_wm1)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt HL
    im_wm1 = DWT_RGB.DWT_RGB_HL_EMBED(im_path, wm_path)
    im_wm1 = cv2.imread(im_wm1, 8)
    ext1 = DWT_RGB.DWT_RGB_HL_EXTRACT(im_path, wm_path, im_wm1)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # svd
    im_wm1 = SVD_RGB.SVD_RGB_EMBED(im_path, wm_path)
    im_wm1 = cv2.imread(im_wm1, 8)
    ext1 = SVD_RGB.SVD_RGB_EXTRACT(im_path, wm_path, im_wm1)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-svd LL
    im_wm1 = DWT_SVD_RGB.DWT_SVD_RGB_LL_EMBED(im_path, wm_path)
    im_wm1 = cv2.imread(im_wm1, 8)
    ext1 = DWT_SVD_RGB.DWT_SVD_RGB_LL_EXTRACT(im_path, wm_path, im_wm1)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-svd HL
    im_wm1 = DWT_SVD_RGB.DWT_SVD_RGB_HL_EMBED(im_path, wm_path)
    im_wm1 = cv2.imread(im_wm1, 8)
    ext1 = DWT_SVD_RGB.DWT_SVD_RGB_HL_EXTRACT(im_path, wm_path, im_wm1)
    psnr1 = Attacks.check_psnr(wm, ext1)
    psnrs.append(convert_psnr_to_latex(psnr1))
    # dwt-dct LL
    # dwt-dct HL
    print(print_latex_format(psnrs))


# ________________ testing playground

# base_psnr_for_attacked()

# ______ DWT
# test_DWT_GRAY_LL()
# test_DWT_GRAY_HL()
# test_DWT_RGB_LL()
# test_DWT_RGB_HL()

# ______ SVD
# test_SVD_GRAY()
# test_SVD_RGB()

# ______ DWT-SVD
# test_DWT_SVD_GRAY_LL()
# test_DWT_SVD_GRAY_HL()
# test_DWT_SVD_RGB_LL()
# test_DWT_SVD_RGB_HL()


test_GRAY()
test_RGB()
