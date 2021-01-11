import numpy as np
import cv2

IMAGES_DIR = "processed_images\\"
SVD_WATERMARKING_CONDITION = 0.01

def SVD_GRAY_EMBED(coverImagePath, watermarkImagePath):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")

    U_cover_img, S_cover_img, V_cover_img = svd(coverImage)
    U_watermark_img, S_watermark_img, V_watermark_img = svd(watermarkImage)

    S_embedded_watermark = embed_watermark_svd(S_cover_img, S_watermark_img )
    watermarked_img = reverse_svd(U_cover_img, S_embedded_watermark, V_cover_img)

    out_path = IMAGES_DIR + 'watermarked_image_SVD_GRAY.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path

def SVD_GRAY_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = read_file(coverImagePath, "GRAY")
    watermark_img = read_file(watermarkImagePath, "GRAY")

    U_cover_img, S_cover_img, V_cover_img = svd(cover_img)
    U_watermark_img, S_watermark_img, V_watermark_img = svd(watermark_img)
    U_watermarked_img, S_watermarked_img, V_watermarked_img = svd(watermarked_img)

    S_extracted_watermark = extract_watermark_svd(S_cover_img, S_watermarked_img)
    extracted_watermark = reverse_svd(U_watermark_img,S_extracted_watermark, V_watermark_img)

    out_path = IMAGES_DIR + 'extracted_watermark_SVD_GRAY.jpg'
    cv2.imwrite(out_path, extracted_watermark)

    return out_path

def svd(band):
    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(band, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(band)))
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

    watermarked_img_gray = SVD_GRAY_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_gray = SVD_GRAY_EXTRACT(coverImagePath, watermarkImagePath, read_file(watermarked_img_gray, "GRAY"))
