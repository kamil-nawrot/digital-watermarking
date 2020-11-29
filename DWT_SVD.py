import numpy as np
import cv2
import pywt
import random
import math
import cmath
from skimage import data
from skimage.color import rgb2gray


def DWT_SVD_RGB_LL(coverImagePath,watermarkImagePath):
    # Cover Image
    # coverImage = cv2.imread('mandrill.jpg', 8)
    coverImage = readFile(coverImagePath,"RGB")
    watermarkImage = readFile(watermarkImagePath,"RGB")

    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # get cover image color chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL,(cr_LH,cr_HL,cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(cr_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cr_LL)))
    np.fill_diagonal(S, S_imgR1)
    S_imgR1 = S
    V_imgR1 = V_imgR1.T.conj()

    # SVD on cover image LL green chanell
    U_imgG1, S_imgG1, V_imgG1 = np.linalg.svd(cg_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cg_LL)))
    np.fill_diagonal(S, S_imgG1)
    S_imgG1 = S
    V_imgG1 = V_imgG1.T.conj()

    # SVD on cover image LL blue chanell
    U_imgB1, S_imgB1, V_imgB1 = np.linalg.svd(cb_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cb_LL)))
    np.fill_diagonal(S, S_imgB1)
    S_imgB1 = S
    V_imgB1 = V_imgB1.T.conj()

    #Watermark Image
    # watermarkImage = cv2.imread('lenna.jpg', 8)
    # cv2.imshow('watermark image', watermarkImage)

    # get color watermark chanels BGR
    watermark_red = watermarkImage[:, :, 2]
    watermark_green = watermarkImage[:, :, 1]
    watermark_blue = watermarkImage[:, :, 0]

    # dwt on watermark on particular color channels
    wr_LL, (wr_LH, wr_HL, wr_HH) = pywt.dwt2(watermark_red, 'haar')
    wg_LL, (wg_LH, wg_HL, wg_HH) = pywt.dwt2(watermark_green, 'haar')
    wb_LL, (wb_LH, wb_HL, wb_HH) = pywt.dwt2(watermark_blue, 'haar')

    # SVD on watermark image LL red chanell
    U_imgR2, S_imgR2, V_imgR2 = np.linalg.svd(wr_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wr_LL)))
    np.fill_diagonal(S, S_imgR2)
    S_imgR2 = S
    V_imgR2 = V_imgR2.T.conj()

    # SVD on cover image LL green chanell
    U_imgG2, S_imgG2, V_imgG2 = np.linalg.svd(wg_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wg_LL)))
    np.fill_diagonal(S, S_imgG2)
    S_imgG2 = S
    V_imgG2 = V_imgG2.T.conj()

    # SVD on cover image LL blue chanell
    U_imgB2, S_imgB2, V_imgB2 = np.linalg.svd(wb_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wb_LL)))
    np.fill_diagonal(S, S_imgB2)
    S_imgB2 = S
    V_imgB2 = V_imgB2.T.conj()


    #Watermarking

    #Embeding algorithm
    S_wimgR = S_imgR1 + (0.01*S_imgR2)
    S_wimgG = S_imgG1 + (0.01*S_imgG2)
    S_wimgB = S_imgB1 + (0.01*S_imgB2)


    # reverse svd
    wimgr = np.dot(U_imgR1, np.dot(S_wimgR, V_imgR1.transpose()))
    wimgg = np.dot(U_imgG1, np.dot(S_wimgG, V_imgG1.transpose()))
    wimgb = np.dot(U_imgB1, np.dot( S_wimgB, V_imgB1.transpose()))

    #idwt for all color channels - reconstruction
    r = pywt.idwt2((wimgr,(cr_LH,cr_HL,cr_HH)), 'haar')
    g = pywt.idwt2((wimgg, (cg_LH, cg_HL, cg_HH)), 'haar')
    b = pywt.idwt2((wimgb, (cb_LH, cb_HL, cb_HH)), 'haar')
    watermarked_img = np.dstack((b,g,r))

    #Show image with embedded watermark
    cv2.imshow('Watermarked Image', np.uint8(watermarked_img))
    cv2.imwrite('watermarked_image_DWT_SVD_RGB_LL.jpg',watermarked_img);

    # Extracting embeded watermark

    #divide watermarked channel for color channels
    cover_red1 = watermarked_img[:, :, 2]
    cover_green1 = watermarked_img[:, :, 1]
    cover_blue1 = watermarked_img[:, :, 0]

    #dwt on watermarked image color channels
    rwed_LL, (rwed_LH, rwed_HL, rwed_HH) = pywt.dwt2(cover_red1, 'haar')
    gwed_LL, (gwed_LH, gwed_HL, gwed_HH) = pywt.dwt2(cover_green1, 'haar')
    bwed_LL, (bwed_LH, bwed_HL, bwed_HH) = pywt.dwt2(cover_blue1, 'haar')

    #svd on watermarked image red channel
    U_imgR_wed, S_imgR_wed, V_imgR_wed = np.linalg.svd(rwed_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(rwed_LL)))
    np.fill_diagonal(S, S_imgR_wed)
    S_imgR_wed = S
    V_imgR_wed = V_imgR_wed.T.conj()

    # svd on watermarked image green channel
    U_imgG_wed, S_imgG_wed, V_imgG_wed = np.linalg.svd(gwed_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(gwed_LL)))
    np.fill_diagonal(S, S_imgG_wed)
    S_imgG_wed = S
    V_imgG_wed = V_imgG_wed.T.conj()

    # svd on watermarked image blue channel
    U_imgB_wed, S_imgB_wed, V_imgB_wed = np.linalg.svd(bwed_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(bwed_LL)))
    np.fill_diagonal(S, S_imgB_wed)
    S_imgB_wed = S
    V_imgB_wed = V_imgB_wed.T.conj()

    # extracting algorithm
    S_ewatr = (S_imgR_wed - S_imgR1) /0.01
    S_ewatg = (S_imgG_wed - S_imgG1) /0.01
    S_ewatb = (S_imgB_wed - S_imgB1) /0.01

    #revert svd on watermarked image color channels
    ewatr = np.dot(U_imgR2, np.dot(S_ewatr, V_imgR2.transpose()))
    ewatg = np.dot(U_imgG2, np.dot(S_ewatg, V_imgG2.transpose()))
    ewatb = np.dot(U_imgB2, np.dot(S_ewatb, V_imgB2.transpose()))

    # idwt on watermarked image color channels
    rw = pywt.idwt2((ewatr, (wr_LH, wr_HL, wr_HH)), 'haar')
    gw = pywt.idwt2((ewatg, (wg_LH, wg_HL, wg_HH)), 'haar')
    bw = pywt.idwt2((ewatb, (wb_LH, wb_HL, wb_HH)), 'haar')

    # compose color channels to BGR image
    extracted_watermark = np.dstack((bw, gw, rw))

    #show extracted image
    cv2.imshow('Extracted Watermark', np.uint8(extracted_watermark))

    cv2.imwrite('extracted_watermark_DWT_SVD_RGB_LL.jpg', extracted_watermark )
    return extracted_watermark

def DWT_SVD_RGB_HL(coverImagePath,watermarkImagePath):
    # Cover Image
    # coverImage = cv2.imread('mandrill.jpg', 8)
    # cv2.imshow('orginal image', coverImage)
    coverImage = readFile(coverImagePath, "RGB")
    watermarkImage = readFile(watermarkImagePath, "RGB")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # get cover image color chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL,(cr_LH,cr_HL,cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(cr_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cr_HL)))
    np.fill_diagonal(S, S_imgR1)
    S_imgR1 = S
    V_imgR1 = V_imgR1.T.conj()

    # SVD on cover image LL green chanell
    U_imgG1, S_imgG1, V_imgG1 = np.linalg.svd(cg_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cg_HL)))
    np.fill_diagonal(S, S_imgG1)
    S_imgG1 = S
    V_imgG1 = V_imgG1.T.conj()

    # SVD on cover image LL blue chanell
    U_imgB1, S_imgB1, V_imgB1 = np.linalg.svd(cb_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(cb_HL)))
    np.fill_diagonal(S, S_imgB1)
    S_imgB1 = S
    V_imgB1 = V_imgB1.T.conj()

    #Watermark Image
    # watermarkImage = cv2.imread('lenna.jpg', 8)
    # cv2.imshow('watermark image', watermarkImage)

    # get color watermark chanels BGR
    watermark_red = watermarkImage[:, :, 2]
    watermark_green = watermarkImage[:, :, 1]
    watermark_blue = watermarkImage[:, :, 0]

    # dwt on watermark on particular color channels
    wr_LL, (wr_LH, wr_HL, wr_HH) = pywt.dwt2(watermark_red, 'haar')
    wg_LL, (wg_LH, wg_HL, wg_HH) = pywt.dwt2(watermark_green, 'haar')
    wb_LL, (wb_LH, wb_HL, wb_HH) = pywt.dwt2(watermark_blue, 'haar')

    # SVD on watermark image LL red chanell
    U_imgR2, S_imgR2, V_imgR2 = np.linalg.svd(wr_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wr_HL)))
    np.fill_diagonal(S, S_imgR2)
    S_imgR2 = S
    V_imgR2 = V_imgR2.T.conj()

    # SVD on cover image LL green chanell
    U_imgG2, S_imgG2, V_imgG2 = np.linalg.svd(wg_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wg_HL)))
    np.fill_diagonal(S, S_imgG2)
    S_imgG2 = S
    V_imgG2 = V_imgG2.T.conj()

    # SVD on cover image LL blue chanell
    U_imgB2, S_imgB2, V_imgB2 = np.linalg.svd(wb_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(wb_HL)))
    np.fill_diagonal(S, S_imgB2)
    S_imgB2 = S
    V_imgB2 = V_imgB2.T.conj()


    #Watermarking

    #Embeding algorithm
    S_wimgR = S_imgR1 + (0.01*S_imgR2)
    S_wimgG = S_imgG1 + (0.01*S_imgG2)
    S_wimgB = S_imgB1 + (0.01*S_imgB2)


    # reverse svd
    wimgr = np.dot(U_imgR1, np.dot(S_wimgR, V_imgR1.transpose()))
    wimgg = np.dot(U_imgG1, np.dot(S_wimgG, V_imgG1.transpose()))
    wimgb = np.dot(U_imgB1, np.dot( S_wimgB, V_imgB1.transpose()))

    #idwt for all color channels - reconstruction
    r = pywt.idwt2((cr_LL,(cr_LH, wimgr, cr_HH)), 'haar')
    g = pywt.idwt2((cg_LL, (cg_LH, wimgg, cg_HH)), 'haar')
    b = pywt.idwt2((cb_LL, (cb_LH, wimgb, cb_HH)), 'haar')
    watermarked_img = np.dstack((b,g,r))

    #Show image with embedded watermark
    cv2.imshow('Watermarked Image', np.uint8(watermarked_img))
    cv2.imwrite('watermarked_image_DWT_SVD_RGB_HL.jpg',watermarked_img);

    # Extracting embeded watermark

    #divide watermarked channel for color channels
    cover_red1 = watermarked_img[:, :, 2]
    cover_green1 = watermarked_img[:, :, 1]
    cover_blue1 = watermarked_img[:, :, 0]

    #dwt on watermarked image color channels
    rwed_LL, (rwed_LH, rwed_HL, rwed_HH) = pywt.dwt2(cover_red1, 'haar')
    gwed_LL, (gwed_LH, gwed_HL, gwed_HH) = pywt.dwt2(cover_green1, 'haar')
    bwed_LL, (bwed_LH, bwed_HL, bwed_HH) = pywt.dwt2(cover_blue1, 'haar')

    #svd on watermarked image red channel
    U_imgR_wed, S_imgR_wed, V_imgR_wed = np.linalg.svd(rwed_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(rwed_HL)))
    np.fill_diagonal(S, S_imgR_wed)
    S_imgR_wed = S
    V_imgR_wed = V_imgR_wed.T.conj()

    # svd on watermarked image green channel
    U_imgG_wed, S_imgG_wed, V_imgG_wed = np.linalg.svd(gwed_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(gwed_HL)))
    np.fill_diagonal(S, S_imgG_wed)
    S_imgG_wed = S
    V_imgG_wed = V_imgG_wed.T.conj()

    # svd on watermarked image blue channel
    U_imgB_wed, S_imgB_wed, V_imgB_wed = np.linalg.svd(bwed_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(bwed_HL)))
    np.fill_diagonal(S, S_imgB_wed)
    S_imgB_wed = S
    V_imgB_wed = V_imgB_wed.T.conj()

    # extracting algorithm
    S_ewatr = (S_imgR_wed - S_imgR1) /0.01
    S_ewatg = (S_imgG_wed - S_imgG1) /0.01
    S_ewatb = (S_imgB_wed - S_imgB1) /0.01

    #revert svd on watermarked image color channels
    ewatr = np.dot(U_imgR2, np.dot(S_ewatr, V_imgR2.transpose()))
    ewatg = np.dot(U_imgG2, np.dot(S_ewatg, V_imgG2.transpose()))
    ewatb = np.dot(U_imgB2, np.dot(S_ewatb, V_imgB2.transpose()))

    # idwt on watermarked image color channels
    rw = pywt.idwt2((wr_LL, (wr_LH, ewatr, wr_HH)), 'haar')
    gw = pywt.idwt2((wr_LL, (wg_LH, ewatg, wg_HH)), 'haar')
    bw = pywt.idwt2((wr_LL, (wb_LH, ewatb, wb_HH)), 'haar')

    # compose color channels to BGR image
    extracted_watermark = np.dstack((bw, gw, rw))

    #show extracted image
    cv2.imshow('Extracted Watermark', np.uint8(extracted_watermark))
    cv2.imwrite('extracted_watermark_DWT_SVD_RGB_HL.jpg',extracted_watermark);

    return extracted_watermark
def DWT_SVD_GRAY_LL(coverImagePath,watermarkImagePath):
    # coverImage = cv2.imread('mandrill.jpg', 0)
    # cv2.imshow('orginal image', coverImage)
    # watermarkImage = cv2.imread('lenna.jpg', 0)
    # cv2.imshow('watermark', watermarkImage)
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)


    #dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    #svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_LL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    #dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    #svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_LL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()

    S_wimg = S_c_img + (0.01 * S_w_img)
    wimgr = np.dot(U_c_img,np.dot( S_wimg, V_c_img.transpose()))
    watermarked_img = pywt.idwt2((wimgr,(c_LH, c_HL, c_HH)), 'haar')
    cv2.imshow('Watermarked Image', np.uint8(watermarked_img))
    cv2.imwrite('watermarked_image_DWT_SVD_GRAY_LL.jpg',watermarked_img);

    #Extracting embeded watermark


    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    U_wed_img, S_wed_img, V_wed_img = np.linalg.svd(wed_LL, full_matrices=1, compute_uv=1)

    S = np.zeros((np.shape(wed_LL)))
    np.fill_diagonal(S, S_wed_img)
    S_wed_img = S
    V_wed_img = V_wed_img.T.conj()

    S_ewat = (S_wed_img - S_c_img) / 0.01
    ewatr = np.dot(U_w_img, np.dot(S_ewat, V_w_img.transpose()))

    extracted_watermark = pywt.idwt2((ewatr, (w_LH, w_HL, w_HH)), 'haar')
    cv2.imshow('Extracted Watermark', np.uint8(extracted_watermark))

    cv2.imwrite('extracted_watermark_DWT_SVD_GRAY_LL.jpg',extracted_watermark);
    return extracted_watermark

def DWT_SVD_GRAY_HL(coverImagePath,watermarkImagePath ):
    # coverImage = cv2.imread('mandrill.jpg', 0)
    # cv2.imshow('orginal image', coverImage)
    # watermarkImage = cv2.imread('lenna.jpg', 0)
    # cv2.imshow('watermark', watermarkImage)
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_HL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    # dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_HL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()

    S_wimg = S_c_img + (0.01 * S_w_img)
    wimgr = np.dot(U_c_img, np.dot(S_wimg, V_c_img.transpose()))
    watermarked_img = pywt.idwt2((c_LL, (c_LH, wimgr, c_HH)), 'haar')
    cv2.imshow('Watermarked Image', np.uint8(watermarked_img))
    cv2.imwrite('watermarked_image_DWT_SVD_GRAY_HL.jpg',watermarked_img);

    # Extracting embeded watermark

    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    U_wed_img, S_wed_img, V_wed_img = np.linalg.svd(wed_HL, full_matrices=1, compute_uv=1)

    S = np.zeros((np.shape(wed_HL)))
    np.fill_diagonal(S, S_wed_img)
    S_wed_img = S
    V_wed_img = V_wed_img.T.conj()

    S_ewat = (S_wed_img - S_c_img) / 0.01
    ewatr = np.dot(U_w_img, np.dot(S_ewat, V_w_img.transpose()))

    extracted_watermark = pywt.idwt2((w_LL, (w_LH, ewatr, w_HH)), 'haar')
    cv2.imshow('Extracted Watermark', np.uint8(extracted_watermark))

    cv2.imwrite('extracted_watermark_DWT_SVD_GRAY_HL.jpg',extracted_watermark);
    return extracted_watermark

def readFile(path, colourType): # colour type == GRAY or RGB
    if colourType == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif colourType == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")
