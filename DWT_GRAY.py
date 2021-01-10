import logging
import cv2
import numpy as np
import pywt

logging.basicConfig(level=logging.DEBUG)

IMAGES_DIR = "processed_images\\"
GRAY_WATERMARKING_CONDITION = 0.1


def DWT_GRAY_LL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = read_file(coverImagePath, "GRAY")
    watermark_img = read_file(watermarkImagePath, "GRAY")

    # DWT on cover image
    cover_ll, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_img, 'haar')

    # DWT on watermark image
    watermark_ll, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermark_img, 'haar')

    # Embedding watermark
    coeffW = (cover_ll + GRAY_WATERMARKING_CONDITION * watermark_ll, (cover_LH, cover_HL, cover_HH))

    watermarked_img = pywt.idwt2(coeffW, 'haar')
    out_path = IMAGES_DIR + 'watermarked_Image_DWT_GRAY_LL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "GRAY")
    watermark_img = read_file(watermarkImagePath, "GRAY")

    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_img, 'haar')
    watermark_LL, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermark_img, 'haar')
    watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = pywt.dwt2(watermarked_img, 'haar')

    extracted_watermark = extract_gray_watermark_LL(cover_LL, watermarked_LL)

    extracted_watermark = pywt.idwt2((extracted_watermark, (watermark_LH, watermark_HL, watermark_HH)), 'haar')
    extracted_watermark = np.uint8(extracted_watermark)

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_GRAY_LL.jpg'
    cv2.imwrite(out_path, extracted_watermark)

    return out_path

def DWT_GRAY_HL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = read_file(coverImagePath, "GRAY")
    watermark_img = read_file(watermarkImagePath, "GRAY")

    # DWT on cover image
    cover_ll, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_img, 'haar')

    # DWT on watermark image
    watermark_ll, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermark_img, 'haar')

    # Embedding watermark
    coeffW = (cover_ll, (cover_LH, cover_HL + GRAY_WATERMARKING_CONDITION * watermark_HL, cover_HH))

    watermarked_img = pywt.idwt2(coeffW, 'haar')
    out_path = IMAGES_DIR + 'watermarked_Image_DWT_GRAY_HL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path

def DWT_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "GRAY")
    watermark_img = read_file(watermarkImagePath, "GRAY")

    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_img, 'haar')
    watermark_LL, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermark_img, 'haar')
    watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = pywt.dwt2(watermarked_img, 'haar')

    extracted_watermark = extract_gray_watermark_LL(cover_HL, watermarked_HL)

    extracted_watermark = pywt.idwt2((watermark_LL, (watermark_LH, extracted_watermark, watermark_HH)), 'haar')
    extracted_watermark = np.uint8(extracted_watermark)

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_GRAY_LL.jpg'
    cv2.imwrite(out_path, extracted_watermark)

    return out_path

def extract_gray_watermark_LL(cover_band, watermarked_band):
    extracted_watermark = (watermarked_band - cover_band) / GRAY_WATERMARKING_CONDITION
    return extracted_watermark


def read_file(path, color):  # color == GRAY or RGB
    if color == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif color == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")


# # Run dwt GRAY_LL, and dwt RGB_LL methods
if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath = 'images\\lenna_512.jpg'
#
#     watermarked_img_gray = DWT_GRAY_LL_EMBED(coverImagePath, watermarkImagePath)
#     extracted_img_gray = DWT_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath,
#                                              read_file(watermarked_img_gray, "GRAY"))

# watermarked_img_gray = DWT_GRAY_HL_EMBED(coverImagePath, watermarkImagePath)
# extracted_img_gray = DWT_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath,
#                                              read_file(watermarked_img_gray, "GRAY"))
