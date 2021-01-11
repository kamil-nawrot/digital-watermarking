import numpy as np
import cv2

IMAGES_DIR = "processed_images\\"
SVD_WATERMARKING_CONDITION = 0.1


def SVD_RGB_EMBED(coverImagePath, watermarkImagePath):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_blue, cover_green, cover_red = get_rgb_channels(cover_img)

    U_cover_red, S_cover_red, V_cover_red = svd_rgb(cover_red)
    U_cover_green, S_cover_green, V_cover_green = svd_rgb(cover_green)
    U_cover_blue, S_cover_blue, V_cover_blue = svd_rgb(cover_blue)

    watermark_red, watermark_green, watermark_blue = get_rgb_channels(watermark_img)

    U_watermark_red, S_watermark_red, V_watermark_red = svd_rgb(watermark_red)
    U_watermark_green, S_watermark_green, V_watermark_green = svd_rgb(watermark_green)
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd_rgb(watermark_blue)

    S_embedded_watermark_red = embed_watermark_svd(S_cover_red, S_watermark_red)
    S_embedded_watermark_green = embed_watermark_svd(S_cover_green, S_watermark_green)
    S_embedded_watermark_blue = embed_watermark_svd(S_cover_blue, S_watermark_blue)

    watermarked_channel_red = reverse_svd(U_cover_red, S_embedded_watermark_red, V_cover_red)
    watermarked_channel_green = reverse_svd(U_cover_green, S_embedded_watermark_green, V_cover_green)
    watermarked_channel_blue = reverse_svd(U_cover_blue, S_embedded_watermark_blue, V_cover_blue)

    watermarked_img = combine_rgb_channels_to_bgr_img(watermarked_channel_red, watermarked_channel_green,
                                                      watermarked_channel_blue)
    out_path = IMAGES_DIR + 'watermarked_image_SVD_RGB.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def SVD_RGB_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "RGB")
    watermark_img = read_file(watermarkImagePath, "RGB")

    cover_red, cover_green, cover_blue = get_rgb_channels(cover_img)

    U_cover_red, S_cover_red, V_cover_red = svd_rgb(cover_red)
    U_cover_green, S_cover_green, V_cover_green = svd_rgb(cover_green)
    U_cover_blue, S_cover_blue, V_cover_blue = svd_rgb(cover_blue)

    watermark_red, watermark_green, watermark_blue = get_rgb_channels(watermark_img)

    U_watermark_red, S_watermark_red, V_watermark_red = svd_rgb(watermark_red)
    U_watermark_green, S_watermark_green, V_watermark_green = svd_rgb(watermark_green)
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd_rgb(watermark_blue)

    watermarked_red, watermarked_green, watermarked_blue = get_rgb_channels(watermarked_img)

    U_watermarked_red, S_watermarked_red, V_watermarked_red = svd_rgb(watermarked_red)
    U_watermarked_green, S_watermarked_green, V_watermarked_green = svd_rgb(watermarked_green)
    U_watermarked_blue, S_watermarked_blue, V_watermarked_blue = svd_rgb(watermarked_blue)

    S_extracted_watermark_red = extract_watermark_svd(S_cover_red, S_watermarked_red)
    S_extracted_watermark_green = extract_watermark_svd(S_cover_green, S_watermarked_green)
    S_extracted_watermark_blue = extract_watermark_svd(S_cover_blue, S_watermarked_blue)

    extracted_watermark_red = reverse_svd(U_watermark_red, S_extracted_watermark_red, V_watermark_red)
    extracted_watermark_green = reverse_svd(U_watermark_green, S_extracted_watermark_green, V_watermark_green)
    extracted_watermark_blue = reverse_svd(U_watermark_blue, S_extracted_watermark_blue, V_watermark_blue)

    extracted_watermark_rgb = combine_rgb_channels_to_bgr_img(extracted_watermark_red, extracted_watermark_green,
                                                              extracted_watermark_blue)

    out_path = IMAGES_DIR + 'extracted_watermark_SVD_RGB.jpg'
    cv2.imwrite(out_path, extracted_watermark_rgb)

    return out_path


def svd_rgb(img):
    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(img, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(img)))
    np.fill_diagonal(S, S_imgR1)
    S_imgR1 = S
    V_imgR1 = V_imgR1.T.conj()
    return U_imgR1, S_imgR1, V_imgR1


def reverse_svd(U_channel, S_channel, V_channel):
    wimgr = np.dot(U_channel, np.dot(S_channel, V_channel.transpose()))
    return wimgr


def embed_watermark_svd(S_cover, S_watermark):
    S_wimgR = S_cover + (S_watermark * SVD_WATERMARKING_CONDITION)
    return S_wimgR


def extract_watermark_svd(S_cover, S_watermarked):
    S_ewatr = (S_watermarked - S_cover) / SVD_WATERMARKING_CONDITION
    return S_ewatr


def combine_rgb_channels_to_bgr_img(red, green, blue):
    bgr_img = np.dstack((red, green, blue))  # BGR format
    return bgr_img


def get_rgb_channels(img_rgb):
    red_channel = img_rgb[:, :, 2]
    green_channel = img_rgb[:, :, 1]
    blue_channel = img_rgb[:, :, 0]
    return red_channel, green_channel, blue_channel


def read_file(path, color):  # color == GRAY or RGB
    if color == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif color == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")


if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath = 'images\\lenna_512.jpg'

    watermarked_img_rgb = SVD_RGB_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_rgb = SVD_RGB_EXTRACT(coverImagePath, watermarkImagePath, read_file(watermarked_img_rgb, "RGB"))
