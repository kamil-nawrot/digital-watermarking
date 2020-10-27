import numpy as np
import cv2
import pywt
import random
import math
import cmath


def DWT_SVD(coverImage, watermarkImage):
    cv2.imshow('Cover Image', coverImage)
    [m, n] = np.shape(coverImage)
    coverImage = np.double(coverImage)
    cv2.imshow('Watermark Image', watermarkImage)
    watermarkImage = np.double(watermarkImage)

    # Applying DWT on cover image and getting four sub-bands
    coverImage = np.float32(coverImage)
    coverImage /= 255;
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC

    # SVD on cA
    uA, wA, vA = np.linalg.svd(cA, full_matrices=1, compute_uv=1)
    [a1, a2] = np.shape(cA)
    WA = np.zeros((a1, a2), np.uint8)
    WA[:a1, :a2] = np.diag(wA)

    # SVD on cH
    uH, wH, vH = np.linalg.svd(cH, full_matrices=1, compute_uv=1)
    [h1, h2] = np.shape(cH)
    WH = np.zeros((h1, h2), np.uint8)
    WH[:h1, :h2] = np.diag(wH)

    # SVD on cV
    uV, wV, vV = np.linalg.svd(cV, full_matrices=1, compute_uv=1)
    [v1, v2] = np.shape(cV)
    WV = np.zeros((v1, v2), np.uint8)
    WV[:v1, :v2] = np.diag(wV)

    # SVD on cD
    uD, wD, vD = np.linalg.svd(cD, full_matrices=1, compute_uv=1)
    [d1, d2] = np.shape(cV)
    WD = np.zeros((d1, d2), np.uint8)
    WD[:d1, :d2] = np.diag(wD)

    # SVD on watermarked image
    uw, ww, vw = np.linalg.svd(watermarkImage, full_matrices=1, compute_uv=1)
    [x, y] = np.shape(watermarkImage)
    WW = np.zeros((x, y), np.uint8)
    WW[:x, :y] = np.diag(ww)

    # Embedding Process
    for i in range(0, x):
        for j in range(0, y):
            WA[i, j] = WA[i, j] + 0.01 * WW[i, j]

    for i in range(0, x):
        for j in range(0, y):
            WV[i, j] = WV[i, j] + 0.01 * WW[i, j]

    for i in range(0, x):
        for j in range(0, y):
            WH[i, j] = WH[i, j] + 0.01 * WW[i, j]

    for i in range(0, x):
        for j in range(0, y):
            WD[i, j] = WD[i, j] + 0.01 * WW[i, j]

    # Inverse of SVD
    cAnew = np.dot(uA, (np.dot(WA, vA)))
    cHnew = np.dot(uH, (np.dot(WH, vH)))
    cVnew = np.dot(uV, (np.dot(WV, vA)))
    cDnew = np.dot(uD, (np.dot(WD, vD)))

    coeff = cAnew, (cHnew, cVnew, cDnew)

    # Inverse DWT to get watermarked image
    watermarkedImage = pywt.idwt2(coeff, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage)