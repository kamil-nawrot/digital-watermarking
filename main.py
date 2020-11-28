import numpy as np
import cv2
import pywt
import random
import math
import cmath

import Attacks as Attacks


def DWT(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage, (500, 500))
    cv2.imshow('Cover Image', coverImage)
    watermarkImage = cv2.resize(watermarkImage, (250, 250))
    cv2.imshow('Watermark Image', watermarkImage)

    # DWT on cover image
    coverImage = np.float32(coverImage)
    coverImage /= 255
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC

    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255

    # Embedding
    coeffW = (cA + 0.1 * watermarkImage, (cH, cV, cD)) #coeffW = (0.4 * cA + 0.1 * watermarkImage, (cH, cV, cD))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage)

    # Extraction
    coeffWM = pywt.dwt2(watermarkedImage, 'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA - cA) / 0.1   #extracted = (hA - 0.4 * cA) / 0.1
    extracted *= 255
    extracted = np.uint8(extracted)
    cv2.imshow('Extracted', extracted)



def SVD(coverImage, watermarkImage):
    cv2.imshow('Cover Image', coverImage)
    [m, n] = np.shape(coverImage)
    coverImage = np.double(coverImage)
    cv2.imshow('Watermark Image', watermarkImage)
    watermarkImage = np.double(watermarkImage)

    # SVD of cover image
    ucvr, wcvr, vtcvr = np.linalg.svd(coverImage, full_matrices=1, compute_uv=1)
    Wcvr = np.zeros((m, n), np.uint8)
    Wcvr[:m, :n] = np.diag(wcvr)
    Wcvr = np.double(Wcvr)
    [x, y] = np.shape(watermarkImage)

    # modifying diagonal component
    for i in range(0, x):
        for j in range(0, y):
            Wcvr[i, j] = (Wcvr[i, j] + 0.01 * watermarkImage[i, j]) / 255

    # SVD of wcvr
    u, w, v = np.linalg.svd(Wcvr, full_matrices=1, compute_uv=1)

    # Watermarked Image
    S = np.zeros((m, n), np.uint8) #change for 512 from 225
    S[:m, :n] = np.diag(w)
    S = np.double(S)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr)) #  np.matmul- function returns the matrix product of two arrays
    wimg = np.double(wimg)
    wimg *= 255
    watermarkedImage = np.zeros(wimg.shape, np.double)
    normalized = cv2.normalize(wimg, watermarkedImage, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow('Watermarked Image', watermarkedImage)


def DWT_SVD(coverImage, watermarkImage):
    cv2.imshow('Cover Image', coverImage)
    [m, n] = np.shape(coverImage)
    coverImage = np.double(coverImage)

    # Applying DWT on cover image and getting four sub-bands
    coverImage = np.float32(coverImage)
    coverImage /= 255;
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC

    p = np.shape(cA)
    print(p)
    # # SVD on cA
    uA, wA, vA = np.linalg.svd(cA, full_matrices=1, compute_uv=1)
    [a1, a2] = np.shape(cA)
    WA = np.zeros((a1, a2), np.uint8)
    WA[:a1, :a2] = np.diag(wA)

    # SVD on cH
    uH, wH, vH = np.linalg.svd(cH, full_matrices=1, compute_uv=1)
    [h1, h2] = np.shape(cH)
    WH = np.zeros((h1, h2), np.uint8)
    WH[:h1, :h2] = np.diag(cH)

    # SVD on cV
    uV, wV, vV = np.linalg.svd(cV, full_matrices=1, compute_uv=1)
    [v1, v2] = np.shape(cV)
    WV = np.zeros((v1, v2), np.uint8)
    WV[:v1, :v2] = np.diag(cV)

    # SVD on cD
    uD, wD, vD = np.linalg.svd(cD, full_matrices=1, compute_uv=1)
    [d1, d2] = np.shape(cV)
    WD = np.zeros((d1, d2), np.uint8)
    WD[:d1, :d2] = np.diag(cD)

    # SVD on watermark image

    watermarkImage = cv2.resize(watermarkImage, p)
    cv2.imshow('Watermark Image', watermarkImage)
    watermarkImage = np.double(watermarkImage)

    uw, ww, vw = np.linalg.svd(watermarkImage, full_matrices=1, compute_uv=1)
    [x, y] = np.shape(watermarkImage) # example: 250,250
    WW = np.zeros((x, y), np.uint8)
    WW[:x, :y] = np.diag(ww)
    print("shape ww: ", np.shape(ww))
    print(ww)

    print("diagonal WW shape: ",np.shape(WW))
    print(WW)

    # Embedding Process for diagonals
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
    #coeff = cv2.resize(cAnew, (256,256)), (cH, cV, cD )
    #coeff = cA, (cv2.resize(cHnew, (256,256)), cV, cD )
    #coeff = cA, (cH, cv2.resize(cVnew, (256,256)), cD )
    #coeff = cA, (cH, cV, cv2.resize(cDnew, (256,256)))

    # Inverse DWT to get watermarked image
    watermarkedImage = pywt.idwt2(coeff, 'haar')
    cv2.imshow('Watermarked Image', watermarkedImage)


#EXTRACTION watermarked img -> 1dwt -> svd (on LL but here is probably on all sub-bands) & original img -> 1dwt -svd (on LL ...) )


    C = pywt.dwt2(coverImage, 'haar')
    shape_LL = C[0].shape # is LL

    Cw = pywt.dwt2(watermarkedImage, 'haar')

    Ucw, Scw, Vcw = np.linalg.svd(Cw[0])
    Uc, Sc, Vc = np.linalg.svd(C[0])
    shape_LL = np.shape(cA)
    Snew = np.zeros((min(shape_LL), min(shape_LL)))

    Uw, Sw, Vw = np.linalg.svd(WA)
    LLnew1 = Uw.dot(np.diag(Scw)).dot(Vw)

    Wdnew = np.zeros((min(shape_LL), min(shape_LL)))

    Scdiag = np.zeros(shape_LL)
    row = min(shape_LL)
    Scdiag[:row, :row] = np.diag(Sc)
    Sc = Scdiag

    alpha = 0.01
    for py in range(0, min(shape_LL)):
        for px in range(0, min(shape_LL)):
            Wdnew[py][px] = (LLnew1[py][px] - Sc[py][px]) / alpha

    # watermark left after svd * Wdnew * right after svd uw * x * yv
    # cAnew = np.dot(uw, (np.dot(Wdnew, vw)))
    #     cv2.imshow('Extracted Watermark', cAnew)
    cv2.imshow('Extracted Watermark', Wdnew)

def compression(qualtiy):
    return Attacks.compression("lenna_256.jpg", qualtiy)

if __name__ == "__main__":
    coverImage = cv2.imread('mandrill.jpg', 0)
    watermarkImage = cv2.imread('lenna_256.jpg', 0)

    options = {1: DWT,
               2: SVD,
               3: DWT_SVD,
               4: compression
               }
    val = int(input('What type of embedding you want to perform?\n1.DWT\n2.SVD\n3.SVD-DWT\n4.Compression\n'))
    if val < 4:
        options[val](coverImage, watermarkImage)
    elif val == 4:
        quality = int(input('Compression quality <0, 100>\n'))
        options[val](quality)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
