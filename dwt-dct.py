import numpy as np
import cv2
import math
import pywt

def convert_image(imagePath, size):
    image = cv2.imread(imagePath)
    image = np.float32(image) / 255
    image = cv2.resize(image, (size, size))
    (b, g, r) = cv2.split(image)
    return (b, g, r)


def calculate_coefficients(image):
    return list(pywt.wavedec2(image, wavelet="haar", level=1))


def apply_dct(baseImage):
    dctImage = np.empty((len(baseImage), len(baseImage)))
    for r in range(0, len(baseImage), 8):
        for p in range(0, len(baseImage), 8):
            block = baseImage[r:r+8, p:p+8]
            dctBlock = cv2.dct(block)
            dctImage[r:r+8, p:p+8] = dctBlock

    return dctImage


def inverse_dct(dctImage):
    idctImage = np.empty((len(dctImage), len(dctImage)))
    for r in range(0, len(dctImage), 8):
        for p in range(0, len(dctImage), 8):
            idctBlock = cv2.idct(dctImage[r:r+8, p:p+8])
            idctImage[r:r+8, p:p+8] = idctBlock
    return idctImage


def embed_watermark(watermarkImage, baseImage):
    watermarkImage = np.ravel(watermarkImage)
    i = 0
    for r in range(0, len(baseImage), 8):
        for p in range(0, len(baseImage), 8):
            if i < len(watermarkImage):
                dctBlock = baseImage[r:r+8, p:p+8]
                dctBlock[5][5] = watermarkImage[i]
                baseImage[r:r+8, p:p+8] = dctBlock
                i += 1
    return baseImage


def retrieve_watermark (dctWatermarkCoeffs, watermark_size):
    watermark = []
    for r in range(0, len(dctWatermarkCoeffs), 8):
        for p in range(0, len(dctWatermarkCoeffs), 8):
            coeffsBlock = dctWatermarkCoeffs[r:r+8, p:p+8]
            watermark.append(coeffsBlock[5][5])

    watermark = np.array(watermark).reshape(watermark_size, watermark_size)
    return watermark


def recover_watermark(image):
    coeffs_watermarked = calculate_coefficients(image)
    dct_coeffs_watermarked = apply_dct(coeffs_watermarked[0])

    watermark = retrieve_watermark(dct_coeffs_watermarked, 128)
    watermark = np.clip(watermark * 255, 0, 255)
    watermark = watermark.astype("uint8")

    # cv2.imshow("Watermark Image", watermark)
    # cv2.imwrite("watermark.jpg", watermark)
    # cv2.waitKey(0)

    return watermark


baseImage = "images/lenna_256.jpg"
watermarkImage = "images/mandrill.jpg"
imgBGR = convert_image(baseImage, 2048)
wtmBGR = convert_image(watermarkImage, 128)

reconstructedImage = []
retrievedWatermark = []
for channel in range(3):
    wtmCoeffs = calculate_coefficients(imgBGR[channel])
    dctImage = apply_dct(wtmCoeffs[0])
    dctImage = embed_watermark(wtmBGR[channel], dctImage)
    wtmCoeffs[0] = inverse_dct(dctImage)

    reconstructed = pywt.waverec2(wtmCoeffs, "haar")
    retrievedWatermark.append(recover_watermark(reconstructed))

    reconstructed = np.clip(reconstructed * 255, 0, 255)
    reconstructed = reconstructed.astype("uint8")
    reconstructedImage.append(reconstructed) 

    # cv2.imshow("Reconstructed Image", reconstructed)
    # cv2.imwrite("watermarked_image.jpg", reconstructed)
    # cv2.waitKey(0)


finalImage = cv2.merge((reconstructedImage[0], reconstructedImage[1], reconstructedImage[2]))
finalWatermark = cv2.merge((retrievedWatermark[0], retrievedWatermark[1], retrievedWatermark[2]))
# finalImage = np.clip(finalImage * 255, 0, 255)
finalImage = finalImage.astype("uint8")
finalWatermark = finalWatermark.astype("uint8")

cv2.imshow('Color Image', finalImage)
cv2.imwrite("watermarked_image.jpg", finalImage)
cv2.imshow('Color Watermark', finalWatermark)
cv2.imwrite("watermark.jpg", finalWatermark)
cv2.waitKey(0)
