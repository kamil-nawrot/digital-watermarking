import cv2
import numpy as np
import pywt

IMAGES_DIR = "processed_images\\"
DWT_SVD_WATERMARKING_CONDITION = 0.1


def extract_watermark_dwt_svd(S_cover_red_channel, S_watermarked_red_channel):
    S_ewatr = (S_watermarked_red_channel - S_cover_red_channel) / DWT_SVD_WATERMARKING_CONDITION
    return S_ewatr


def embed_watermark_dwt_svd(S_cover_red_channel, S_watermark_red):
    S_wimgR = S_cover_red_channel + (DWT_SVD_WATERMARKING_CONDITION * S_watermark_red)
    return S_wimgR


def svd(subband):
    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(subband, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(subband)))
    np.fill_diagonal(S, S_imgR1)
    S_imgR1 = S
    V_imgR1 = V_imgR1.T.conj()
    return U_imgR1, S_imgR1, V_imgR1


def reverse_svd(U_channel, S_channel, V_channel):
    wimgr = np.dot(U_channel, np.dot(S_channel, V_channel.transpose()))
    return wimgr


def DWT_SVD_GRAY_LL_EMBED(coverImagePath, watermarkImagePath):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")

    # dwt on cover image
    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_cover_img, S_cover_img, V_cover_img = svd(cover_LL)

    # dwt on watermark image
    watermark_LL, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_watermark_img, S_watermark_img, V_watermark_img = svd(watermark_LL)

    S_watermarked = embed_watermark_dwt_svd(S_cover_img, S_watermark_img)

    watermarked_LL = reverse_svd(U_cover_img, S_watermarked, V_cover_img)

    watermarked_img = pywt.idwt2((watermarked_LL, (cover_LH, cover_HL, cover_HH)), 'haar')

    out_path = IMAGES_DIR + 'watermarked_image_DWT_SVD_GRAY_LL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_SVD_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")

    # dwt on cover image
    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_cover_img, S_cover_img, V_cover_img = svd(cover_LL)

    # dwt on watermark image
    watermark_LL, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_watermark_img, S_watermark_img, V_watermark_img = svd(watermark_LL)

    # dwt on watermarked image
    watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = pywt.dwt2(watermarked_img, 'haar')
    # svd on watermarked image
    U_watermarked_img, S_watermarked_img, V_watermarked_img = svd(watermarked_LL);

    # Extracting watermark
    S_watermarked_LL = extract_watermark_dwt_svd(S_cover_img, S_watermarked_img)
    # reverse svd
    watermarked_LL = reverse_svd(U_watermark_img, S_watermarked_LL, V_watermark_img)
    # reverse dwt
    extracted_watermark = pywt.idwt2((watermarked_LL, (watermark_LH, watermark_HL, watermark_HH)), 'haar')

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_SVD_GRAY_LL.jpg'
    cv2.imwrite(out_path, extracted_watermark);

    return out_path


def DWT_SVD_GRAY_HL_EMBED(coverImagePath, watermarkImagePath):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")

    # dwt on cover image
    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_cover_img, S_cover_img, V_cover_img = svd(cover_HL)

    # dwt on watermark image
    watermark_LL, (watermark_LH, watermark_HL, watermark_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_watermark_img, S_watermark_img, V_watermark_img = svd(watermark_HL)

    # Embed watermark
    S_watermarked = embed_watermark_dwt_svd(S_cover_img, S_watermark_img)

    # reverse svd
    watermarked_HL = reverse_svd(U_cover_img, S_watermarked, V_cover_img)

    # reverse dwt
    watermarked_img = pywt.idwt2((cover_LL, (cover_LH, watermarked_HL, cover_HH)), 'haar')

    out_path = IMAGES_DIR + 'watermarked_image_DWT_SVD_GRAY_HL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_SVD_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")

    # dwt on cover image
    cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image HL
    U_cover_img, S_cover_img, V_cover_img = svd(cover_HL)

    # dwt on watermark image
    watermark_LL, (watermark_LH, extracted_watermark_HL, watermark_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image HL
    U_watermark_img, S_watermark_img, V_watermark_img = svd(extracted_watermark_HL)

    # dwt on watermarked image
    watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = pywt.dwt2(watermarked_img, 'haar')
    # svd on watermarked image
    U_watermarked_img, S_watermrked_img, V_watermarked_img = svd(watermarked_HL)

    # Extract watermark
    S_extracted_watermark_HL = extract_watermark_dwt_svd(S_cover_img, S_watermrked_img)

    # reverse svd
    extracted_watermark_HL = np.dot(U_watermark_img, np.dot(S_extracted_watermark_HL, V_watermark_img.transpose()))

    # reverse dwt
    extracted_watermark = pywt.idwt2((watermark_LL, (watermark_LH, extracted_watermark_HL, watermark_HH)), 'haar')

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_SVD_GRAY_HL.jpg'
    cv2.imwrite(out_path, extracted_watermark);
    return out_path


def read_file(path, colourType):  # colour type == GRAY or RGB
    if colourType == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif colourType == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")


# Run all embedding and extraction methods
if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath = 'images\\lenna_512.jpg'

    watermarked_img_gray_hl = DWT_SVD_GRAY_HL_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_gray_hl = DWT_SVD_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath,
                                                    read_file(watermarked_img_gray_hl, "GRAY"))

    watermarked_img_gray_ll = DWT_SVD_GRAY_LL_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_gray_ll = DWT_SVD_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath,
                                                    read_file(watermarked_img_gray_ll, "GRAY"))
