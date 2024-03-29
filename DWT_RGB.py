import logging
import cv2
import numpy as np
import pywt

logging.basicConfig(level=logging.DEBUG)

IMAGES_DIR = "processed_images\\"
RGB_WATERMARKING_CONDITION = 0.10


def DWT_RGB_LL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_red_dwt_layers, cover_green_dwt_layers, cover_blue_dwt_layers = dwt_rgb_image(cover_img)

    watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers = dwt_rgb_image(watermark_img)

    watermarked_cover_img_LL = embed_watermark_to_LL_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                         cover_blue_dwt_layers, watermark_red_dwt_layers,
                                                         watermark_green_dwt_layers,
                                                         watermark_blue_dwt_layers)

    cover_red_dwt_layers[0] = watermarked_cover_img_LL[0]
    cover_green_dwt_layers[0] = watermarked_cover_img_LL[1]
    cover_blue_dwt_layers[0] = watermarked_cover_img_LL[2]

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                               cover_blue_dwt_layers)

    watermarked_img = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'watermarked_image_DWT_RGB_LL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_red_dwt_layers, cover_green_dwt_layers, cover_blue_dwt_layers = dwt_rgb_image(cover_img)

    watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers = dwt_rgb_image(watermark_img)

    watermarked_red_layers, watermarked_green_layers, watermarked_blue_layers = dwt_rgb_image(watermarked_img)

    extracted_watermark_ll = extract_watermark_from_LL_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                           cover_blue_dwt_layers,
                                                           watermarked_red_layers, watermarked_green_layers,
                                                           watermarked_blue_layers)
    watermark_red_dwt_layers[0] = extracted_watermark_ll[0]
    watermark_green_dwt_layers[0] = extracted_watermark_ll[1]
    watermark_blue_dwt_layers[0] = extracted_watermark_ll[2]

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(
        watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers
    )

    extracted_watermark = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_RBG_LL.jpg'
    cv2.imwrite(out_path, extracted_watermark)

    return out_path


def DWT_RGB_HL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_red_dwt_layers, cover_green_dwt_layers, cover_blue_dwt_layers = dwt_rgb_image(cover_img)

    watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers = dwt_rgb_image(watermark_img)

    watermarked_cover_img_HL = embed_watermark_to_HL_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                         cover_blue_dwt_layers, watermark_red_dwt_layers,
                                                         watermark_green_dwt_layers,
                                                         watermark_blue_dwt_layers)

    cover_red_dwt_layers[2] = watermarked_cover_img_HL[0]
    cover_green_dwt_layers[2] = watermarked_cover_img_HL[1]
    cover_blue_dwt_layers[2] = watermarked_cover_img_HL[2]

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                               cover_blue_dwt_layers)

    watermarked_img = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'watermarked_image_DWT_RGB_HL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path

def DWT_RGB_HL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_red_dwt_layers, cover_green_dwt_layers, cover_blue_dwt_layers = dwt_rgb_image(cover_img)

    watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers = dwt_rgb_image(watermark_img)

    watermarked_red_layers, watermarked_green_layers, watermarked_blue_layers = dwt_rgb_image(watermarked_img)

    extracted_watermark_HL = extract_watermark_from_HL_rgb(cover_red_dwt_layers, cover_green_dwt_layers,
                                                           cover_blue_dwt_layers,
                                                           watermarked_red_layers, watermarked_green_layers,
                                                           watermarked_blue_layers)
    watermark_red_dwt_layers[2] = extracted_watermark_HL[0]
    watermark_green_dwt_layers[2] = extracted_watermark_HL[1]
    watermark_blue_dwt_layers[2] = extracted_watermark_HL[2]

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(
        watermark_red_dwt_layers, watermark_green_dwt_layers, watermark_blue_dwt_layers
    )

    extracted_watermark = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_RBG_HL.jpg'
    cv2.imwrite(out_path, extracted_watermark)

    return out_path



def embed_watermark_to_LL_rgb(cover_red_layers, cover_green_layers, cover_blue_layers, watermark_red, watermark_green,
                              watermark_blue):
    watermarked_red = cover_red_layers[0] + RGB_WATERMARKING_CONDITION * watermark_red[0]
    watermarked_green = cover_green_layers[0] + RGB_WATERMARKING_CONDITION * watermark_green[0]
    watermarked_blue = cover_blue_layers[0] + RGB_WATERMARKING_CONDITION * watermark_blue[0]
    return watermarked_red, watermarked_green, watermarked_blue

def embed_watermark_to_HL_rgb(cover_red_layers, cover_green_layers, cover_blue_layers, watermark_red, watermark_green,
                              watermark_blue):
    watermarked_red = cover_red_layers[2] + RGB_WATERMARKING_CONDITION * watermark_red[2]
    watermarked_green = cover_green_layers[2] + RGB_WATERMARKING_CONDITION * watermark_green[2]
    watermarked_blue = cover_blue_layers[2] + RGB_WATERMARKING_CONDITION * watermark_blue[2]
    return watermarked_red, watermarked_green, watermarked_blue


def reverse_dwt_rgb(watermark_red, watermark_green, watermark_blue):
    red_channel = pywt.idwt2(
        (watermark_red[0], (watermark_red[1], watermark_red[2], watermark_red[3])), 'haar')
    green_channel = pywt.idwt2(
        (watermark_green[0], (watermark_green[1], watermark_green[2], watermark_green[3])), 'haar')
    blue_channel = pywt.idwt2(
        (watermark_blue[0], (watermark_blue[1], watermark_blue[2], watermark_blue[3])), 'haar')
    return red_channel, green_channel, blue_channel


def combine_rgb_channels_to_bgr_img(red, green, blue):
    bgr_img = np.dstack((blue, green, red))  # BGR format
    return bgr_img


def dwt_rgb_image(coverImage):
    # get color cover chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]
    # dwt on cover image on particular color channels
    cr_LL, (cr_LH, cr_HL, cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')
    return [cr_LL, cr_LH, cr_HL, cr_HH], [cg_LL, cg_LH, cg_HL, cg_HH], [cb_LL, cb_LH, cb_HL, cb_HH]


def extract_watermark_from_LL_rgb(cover_red_layers, cover_green_layers, cover_blue_layers, watermarked_red,
                                  watermarked_green, watermarked_blue):
    extracted_watermark_red = (watermarked_red[0] - cover_red_layers[0]) / RGB_WATERMARKING_CONDITION
    extracted_watermark_green = (watermarked_green[0] - cover_green_layers[0]) / RGB_WATERMARKING_CONDITION
    extracted_watermark_blue = (watermarked_blue[0] - cover_blue_layers[0]) / RGB_WATERMARKING_CONDITION
    return extracted_watermark_red, extracted_watermark_green, extracted_watermark_blue

def extract_watermark_from_HL_rgb(cover_red_layers, cover_green_layers, cover_blue_layers, watermarked_red,
                                  watermarked_green, watermarked_blue):
    extracted_watermark_red = (watermarked_red[2] - cover_red_layers[2]) / RGB_WATERMARKING_CONDITION
    extracted_watermark_green = (watermarked_green[2] - cover_green_layers[2]) / RGB_WATERMARKING_CONDITION
    extracted_watermark_blue = (watermarked_blue[2] - cover_blue_layers[2]) / RGB_WATERMARKING_CONDITION
    return extracted_watermark_red, extracted_watermark_green, extracted_watermark_blue


def read_file(path, color):  # color == GRAY or RGB
    if color == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif color == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")


# Run rgb tests
if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath = 'images\\lenna_512.jpg'

    watermarked_img_rbg = DWT_RGB_LL_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_rbg = DWT_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath, read_file(watermarked_img_rbg, "RGB"))

    watermarked_img_rbg = DWT_RGB_HL_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_rbg = DWT_RGB_HL_EXTRACT(coverImagePath, watermarkImagePath, read_file(watermarked_img_rbg, "RGB"))
