import numpy as np
import cv2
import math
import pywt

def convert_image(imagePath, size):
    image = cv2.imread(imagePath, flags=cv2.IMREAD_GRAYSCALE)
    image = np.float32(image) / 255
    image = cv2.resize(image, (size, size))
    return image


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

    watermark = retrieve_watermark(dct_coeffs_watermarked, 64)
    watermark = np.clip(watermark * 255, 0, 255)
    watermark = watermark.astype("uint8")

    cv2.imshow("Watermark Image", watermark)
    cv2.imwrite("watermark.jpg", watermark)
    cv2.waitKey(0)


baseImage = "images/lenna_256.jpg"
watermarkImage = "images/mandrill_256.jpg"
img = convert_image(baseImage, 1024)
wtm = convert_image(watermarkImage, 64)

wtmCoeffs = calculate_coefficients(img)
dctImage = apply_dct(wtmCoeffs[0])
dctImage = embed_watermark(wtm, dctImage)
wtmCoeffs[0] = inverse_dct(dctImage)

reconstructedImage = pywt.waverec2(wtmCoeffs, "haar")
recover_watermark(reconstructedImage)

reconstructedImage = np.clip(reconstructedImage * 255, 0, 255)
reconstructedImage = reconstructedImage.astype("uint8")

cv2.imshow("Reconstructed Image", reconstructedImage)
cv2.imwrite("watermarked_image.jpg", reconstructedImage)
cv2.waitKey(0)
