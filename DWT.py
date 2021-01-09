import cv2
import numpy as np
import pywt
import logging
logging.basicConfig(level=logging.DEBUG)

IMAGES_DIR = "processed_images\\"

def DWT_RGB_LL(coverImagePath, watermarkImagePath):
    coverImage = read_file(coverImagePath, "RGB")
    watermarkImage = read_file(watermarkImagePath, "RGB")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # get color cover chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL, (cr_LH, cr_HL, cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # get color w   atermark chanels BGR
    watermark_red = watermarkImage[:, :, 2]
    watermark_green = watermarkImage[:, :, 1]
    watermark_blue = watermarkImage[:, :, 0]

    # dwt on watermark on particular color channels
    wr_LL, (wr_LH, wr_HL, wr_HH) = pywt.dwt2(watermark_red, 'haar')
    wg_LL, (wg_LH, wg_HL, wg_HH) = pywt.dwt2(watermark_green, 'haar')
    wb_LL, (wb_LH, wb_HL, wb_HH) = pywt.dwt2(watermark_blue, 'haar')

    # Embedding
    red_LL_embeded = cr_LL + 0.01 * wr_LL
    green_LL_embeded = cg_LL + 0.01 * wg_LL
    blue_LL_embeded = cb_LL + 0.01 * wb_LL

    r = pywt.idwt2((red_LL_embeded, (cr_LH, cr_HL, cr_HH)), 'haar')
    g = pywt.idwt2((green_LL_embeded, (cg_LH, cg_HL, cg_HH)), 'haar')
    b = pywt.idwt2((blue_LL_embeded, (cb_LH, cb_HL, cb_HH)), 'haar')
    watermarked_img = np.dstack((b, g, r))

    cv2.imshow('Watermarked Image', watermarked_img.astype(np.uint8))
    cv2.imwrite(''+IMAGES_DIR +'watermarked_image_DWT_RGB_LL.jpg', watermarked_img)

    return watermarked_img

def DWT_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = read_file(coverImagePath, "RGB")
    watermarkImage = read_file(watermarkImagePath, "RGB")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # get color cover chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL, (cr_LH, cr_HL, cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # get color w   atermark chanels BGR
    watermark_red = watermarkImage[:, :, 2]
    watermark_green = watermarkImage[:, :, 1]
    watermark_blue = watermarkImage[:, :, 0]

    # dwt on watermark on particular color channels
    wr_LL, (wr_LH, wr_HL, wr_HH) = pywt.dwt2(watermark_red, 'haar')
    wg_LL, (wg_LH, wg_HL, wg_HH) = pywt.dwt2(watermark_green, 'haar')
    wb_LL, (wb_LH, wb_HL, wb_HH) = pywt.dwt2(watermark_blue, 'haar')

    cover_red1 = watermarked_img[:, :, 2]
    cover_green1 = watermarked_img[:, :, 1]
    cover_blue1 = watermarked_img[:, :, 0]

    # dwt on watermarked image color channels
    rwed_LL, (rwed_LH, rwed_HL, rwed_HH) = pywt.dwt2(cover_red1, 'haar')
    gwed_LL, (gwed_LH, gwed_HL, gwed_HH) = pywt.dwt2(cover_green1, 'haar')
    bwed_LL, (bwed_LH, bwed_HL, bwed_HH) = pywt.dwt2(cover_blue1, 'haar')

    #Extract algorithm on color channels

    extracted_watermark_LL_r = (rwed_LL - cr_LL) / 0.01
    extracted_watermark_LL_g = (gwed_LL - cg_LL) / 0.01
    extracted_watermark_LL_b = (bwed_LL - cb_LL) / 0.01

    # idwt on color channels - getting waermark image
    rw = pywt.idwt2((extracted_watermark_LL_r, (wr_LH, wr_HL, wr_HH)), 'haar')
    gw = pywt.idwt2((extracted_watermark_LL_g, (wg_LH, wg_HL, wg_HH)), 'haar')
    bw = pywt.idwt2((extracted_watermark_LL_b, (wb_LH, wb_HL, wb_HH)), 'haar')

    extracted_watermark = np.dstack((bw, gw, rw))

    cv2.imshow('Extracted watermark', extracted_watermark.astype(np.uint8))
    cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_RBG_LL.jpg', extracted_watermark)

    return extracted_watermark

def DWT_GRAY_LL(coverImagePath, watermarkImagePath):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # DWT on cover image
    coeffC = pywt.dwt2(coverImage, 'haar')
    cr_LL, (cr_LH, cr_HL, cr_HH) = coeffC

    # DWT on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')

    # Embedding watermark
    coeffW = (cr_LL + 0.1 * w_LL, (cr_LH, cr_HL, cr_HH))
    watermarked_img = pywt.idwt2(coeffW, 'haar')

    cv2.imshow('Watermarked Image', watermarked_img)
    cv2.imwrite('' + IMAGES_DIR + 'Watermarked Image_DWT_GRAY_LL.jpg', watermarked_img)

    return watermarked_img


def DWT_GRAY_LL_Extract(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # DWT on cover image
    coeffC = pywt.dwt2(coverImage, 'haar')
    cr_LL, (cr_LH, cr_HL, cr_HH) = coeffC

    # DWT on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')

    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    extracted = (wed_LL - cr_LL) / 0.1   #extracted = (hA - 0.4 * cA) / 0.1
    # extracted *= 255
    extracted_watermark = pywt.idwt2((extracted, (w_LH, w_HL, w_HH)), 'haar')
    extracted_watermark = np.uint8(extracted_watermark)

    cv2.imshow('Extracted', extracted_watermark)
    cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_GRAY_LL.jpg', extracted_watermark)


def read_file(path, color): # color == GRAY or RGB
    if color == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif color == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")

#TESTS


if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath= 'images\\lenna_512.jpg'

    watermarked_img_gray = DWT_GRAY_LL(coverImagePath, watermarkImagePath)
    extracted_img_gray = DWT_GRAY_LL_Extract(coverImagePath, watermarkImagePath, watermarked_img_gray)

    watermarked_img_rbg = DWT_RGB_LL(coverImagePath, watermarkImagePath)
    extracted_img_rbg = DWT_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img_rbg)

