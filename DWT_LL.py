import cv2
import numpy as np
import pywt

def DWT_GRAY_LL(coverImagePath, watermarkImagePath):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

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
def DWT_GRAY_EXTRACT(coverImagePath, watermarkImagePath, watermarkedImagePath):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")
    watermarkedImage = readFile(watermarkedImagePath, "GRAY")

    # cv2.imshow('orginal image', coverImage)
    # cv2.imshow('watermark', watermarkImage)
    # cv2.imshow('watermarked image', watermarkedImage)
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


def readFile(path, colourType):  # colour type == GRAY or RGB
    if colourType == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif colourType == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")