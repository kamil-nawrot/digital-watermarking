import numpy as np
import cv2
import pywt

def DWT_RGB_LL():
    coverImage = cv2.imread('mandrill.jpg', 8)
    cv2.imshow('orginal image', coverImage)
    watermarkImage = cv2.imread('lenna.jpg', 8)
    cv2.imshow('watermark', watermarkImage)

    # get color cover chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL, (cr_LH, cr_HL, cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # get color watermark chanels BGR
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
    cv2.imwrite('watermarked_image_DWT_RGB_LL.jpg', watermarked_img)

    # # Extraction

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

    cv2.imshow('Extracted', extracted_watermark.astype(np.uint8))
    cv2.imwrite('extracted_watermark_DWT_RBG_LL.jpg', extracted_watermark)

def DWT_GRAY_LL():
    coverImage = cv2.imread('mandrill.jpg', 0)
    cv2.imshow('orginal image', coverImage)
    watermarkImage = cv2.imread('lenna.jpg', 0)
    cv2.imshow('watermark', watermarkImage)

    # DWT on cover image

    coeffC = pywt.dwt2(coverImage, 'haar')
    cr_LL, (cr_LH, cr_HL, cr_HH) = coeffC

    # DWT on watermark image

    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')

    # Embedding
    coeffW = (cr_LL + 0.01 * w_LL, (cr_LH, cr_HL, cr_HH))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage.astype(np.uint8))
    cv2.imwrite('watermarked_image_DWT_GRAY_LL.jpg', watermarkedImage)

    # Extraction
    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarkedImage, 'haar')
    extracted = (wed_LL - cr_LL) / 0.01   #extracted = (hA - 0.4 * cA) / 0.1
    # extracted *= 255
    extracted_watermark = pywt.idwt2((extracted, (w_LH, w_HL, w_HH)), 'haar')
    extracted_watermark = np.uint8(extracted_watermark)

    cv2.imshow('Extracted', extracted_watermark)
    cv2.imwrite('extracted_watermark_DWT_GRAY_LL.jpg', extracted_watermark)

#Working on that
def DWT_GRAY_EXTRACT():
     coverImage = cv2.imread('mandrill.jpg', 0)
    # cv2.imshow('orginal image', coverImage)
    # watermarkImage = cv2.imread('lenna.jpg', 0)
    # cv2.imshow('watermark', watermarkImage)
    # watermarkedImage = cv2.imread('extracted_watermark_DWT_GRAY_LL.jpg', 0)
    # cv2.imshow('watermarked image', watermarkedImage)
    #
    # # DWT on cover image
    # coeffC = pywt.dwt2(coverImage, 'haar')
    # cr_LL, (cr_LH, cr_HL, cr_HH) = coeffC
    # # DWT on watermark image
    # w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # #DWT on watermarked image
    # wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarkedImage, 'haar')
    # #Extraction algorithm
    # extracted = (wed_LL - cr_LL) / 0.01  # extracted = (hA - 0.4 * cA) / 0.1
    # # extracted *= 255
    # extracted_watermark = pywt.idwt2((extracted, (w_LH, w_HL, w_HH)), 'haar')
    # extracted_watermark = np.uint8(extracted_watermark)
    #
    # cv2.imshow('Extracted', extracted_watermark)
    # cv2.imwrite('extracted_watermark_DWT_GRAY_LL.jpg', extracted_watermark)

if __name__ == "__main__":

    options = {1: DWT_GRAY_LL, 2: DWT_RGB_LL
               }
    val = int(input('What type of embedding you want to perform?\n1.DWT_GRAY_EMBED\n2.DWT_RGB_LL '))
    options[val]()

    cv2.waitKey(0)
    cv2.destroyAllWindows()