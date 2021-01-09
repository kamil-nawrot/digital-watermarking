import cv2
import numpy as np
import pywt

IMAGES_DIR = "processed_images\\"
DWT_SVD_WATERMARKING_CONDITION = 0.01


def DWT_SVD_RGB_LL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = readFile(coverImagePath, "RGB")
    watermark_img = readFile(watermarkImagePath, "RGB")

    cover_red_dwt_subbands, cover_green_dwt_subbands, cover_blue_dwt_subbands = dwt_rgb_image(cover_img)

    U_cover_red_channel, S_cover_red_channel, V_cover_red_channel = svd(cover_red_dwt_subbands[0])
    U_cover_green_channel, S_cover_green_channel, V_cover_green_channel = svd(cover_green_dwt_subbands[0])
    U_cover_blue_channel, S_cover_blue_channel, V_cover_blue_channel = svd(cover_blue_dwt_subbands[0])

    watermark_red_dwt_subbands, watermark_green_dwt_subbands, watermark_blue_dwt_subbands = dwt_rgb_image(watermark_img)

    U_watermark_red, S_watermark_red, V_watermark_red = svd(watermark_red_dwt_subbands[0])
    U_watermark_green, S_watermark_green, V_watermark_green = svd(
        watermark_green_dwt_subbands[0])
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd(
        watermark_blue_dwt_subbands[0])

    # Embeding algorithm
    S_watermarked_red = embed_watermark_dwt_svd(S_cover_red_channel, S_watermark_red)
    S_watermarked_green = embed_watermark_dwt_svd(S_cover_green_channel, S_watermark_green)
    S_watermarked_blue = embed_watermark_dwt_svd(S_cover_blue_channel, S_watermark_blue)

    # reverse svd
    watermarked_img_red = reverse_svd(U_cover_red_channel, S_watermarked_red, V_cover_red_channel)
    watermarked_img_green = reverse_svd(U_cover_green_channel, S_watermarked_green, V_cover_green_channel)
    watermarked_img_blue = reverse_svd(U_cover_blue_channel, S_watermarked_blue, V_cover_blue_channel)

    # replace cover_img_LL subband with watermarked LL subband
    cover_red_dwt_subbands[0] = watermarked_img_red
    cover_green_dwt_subbands[0] = watermarked_img_green
    cover_blue_dwt_subbands[0] = watermarked_img_blue

    # reverse dwt on watermarked cover image
    red_channel, green_channel, blue_channel = reverse_dwt_rgb(cover_red_dwt_subbands, cover_green_dwt_subbands,
                                                               cover_blue_dwt_subbands)

    watermarked_img = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    # Show image with embedded watermark

    out_path = IMAGES_DIR + 'watermarked_image_DWT_SVD_RGB_LL.jpg'
    cv2.imwrite(out_path, watermarked_img)
    return out_path


def embed_watermark_dwt_svd(S_cover_red_channel, S_watermark_red):
    S_wimgR = S_cover_red_channel + (DWT_SVD_WATERMARKING_CONDITION * S_watermark_red)
    return S_wimgR


def reverse_svd(U_channel, S_channel, V_channel):
    wimgr = np.dot(U_channel, np.dot(S_channel, V_channel.transpose()))
    return wimgr


def svd(subband):
    # SVD on cover image LL red chanell
    U_imgR1, S_imgR1, V_imgR1 = np.linalg.svd(subband, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(subband)))
    np.fill_diagonal(S, S_imgR1)
    S_imgR1 = S
    V_imgR1 = V_imgR1.T.conj()
    return U_imgR1, S_imgR1, V_imgR1


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


def reverse_dwt_rgb(watermark_red_dwt_subbands, watermark_green_dwt_subbands, watermark_blue_dwt_subbands):
    red_channel = pywt.idwt2(
        (watermark_red_dwt_subbands[0],
         (watermark_red_dwt_subbands[1], watermark_red_dwt_subbands[2], watermark_red_dwt_subbands[3])), 'haar')
    green_channel = pywt.idwt2(
        (watermark_green_dwt_subbands[0],
         (watermark_green_dwt_subbands[1], watermark_green_dwt_subbands[2], watermark_green_dwt_subbands[3])), 'haar')
    blue_channel = pywt.idwt2(
        (watermark_blue_dwt_subbands[0],
         (watermark_blue_dwt_subbands[1], watermark_blue_dwt_subbands[2], watermark_blue_dwt_subbands[3])), 'haar')
    return red_channel, green_channel, blue_channel


def combine_rgb_channels_to_bgr_img(red, green, blue):
    bgr_img = np.dstack((blue, green, red))  # BGR format
    return bgr_img


def DWT_SVD_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = readFile(coverImagePath, "RGB")
    watermark_img = readFile(watermarkImagePath, "RGB")

    cover_red_dwt_subbands, cover_green_dwt_subbands, cover_blue_dwt_subbands = dwt_rgb_image(cover_img)

    U_cover_red_channel, S_cover_red_channel, V_cover_red_channel = svd(cover_red_dwt_subbands[0])
    U_cover_green_channel, S_cover_green_channel, V_cover_green_channel = svd(cover_green_dwt_subbands[0])
    U_cover_blue_channel, S_cover_blue_channel, V_cover_blue_channel = svd(cover_blue_dwt_subbands[0])

    # Watermark Image

    watermark_red_dwt_subbands, watermark_green_dwt_subbands, watermark_blue_dwt_subbands = dwt_rgb_image(watermark_img)

    U_watermark_red, S_watermark_red, V_watermark_red = svd(watermark_red_dwt_subbands[0])
    U_watermark_green, S_watermark_green, V_watermark_green = svd(
        watermark_green_dwt_subbands[0])
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd(
        watermark_blue_dwt_subbands[0])

    # dwt on watermarked image
    watermarked_red_dwt_subbands, watermarked_green_dwt_subbands, watermarked_blue_dwt_subbands = dwt_rgb_image(
        watermarked_img)
    # svd on watermarked LL subband image
    U_watermarked_red_channel, S_watermarked_red_channel, V_watermarked_red_channel = svd(
        watermarked_red_dwt_subbands[0])
    U_watermarked_green_channel, S_watermarked_green_channel, V_watermarked_green_channel = svd(
        watermarked_green_dwt_subbands[0])
    U_watermarked_blue_channel, S_watermarked_blue_channel, V_watermarked_blue_channel = svd(
        watermarked_blue_dwt_subbands[0])

    # extracting algorithm
    S_extracted_watermark_red = extract_watermark_dwt_svd(S_cover_red_channel, S_watermarked_red_channel)
    S_extracted_watermark_green = extract_watermark_dwt_svd(S_cover_green_channel, S_watermarked_green_channel)
    S_extracted_watermark_blue = extract_watermark_dwt_svd(S_cover_blue_channel, S_watermarked_blue_channel)

    # revert svd on watermarked image channels. Receives subband LL
    extracted_watermark_red_LL = reverse_svd(U_watermark_red, S_extracted_watermark_red, V_watermark_red)
    extracted_watermark_green_LL = reverse_svd(U_watermark_green, S_extracted_watermark_green, V_watermark_green)
    extracted_watermark_blue_LL = reverse_svd(U_watermark_blue, S_extracted_watermark_blue, V_watermark_blue)

    # replace watermark subband with watermarked LL subbandr
    watermark_red_dwt_subbands[0] = extracted_watermark_red_LL
    watermark_green_dwt_subbands[0] = extracted_watermark_green_LL
    watermark_blue_dwt_subbands[0] = extracted_watermark_blue_LL

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(watermark_red_dwt_subbands, watermark_green_dwt_subbands,
                                                               watermark_blue_dwt_subbands)

    # compose color channels to BGR image
    extracted_watermark = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    cv2.imwrite(out_path, extracted_watermark)
    return out_path


out_path = IMAGES_DIR + 'extracted_watermark_DWT_SVD_RGB_LL.jpg'

def extract_watermark_dwt_svd(S_cover_red_channel, S_watermarked_red_channel):
    S_ewatr = (S_watermarked_red_channel - S_cover_red_channel) / DWT_SVD_WATERMARKING_CONDITION
    return S_ewatr

def DWT_SVD_RGB_HL_EMBED(coverImagePath, watermarkImagePath):
    cover_img = readFile(coverImagePath, "RGB")
    watermark_img = readFile(watermarkImagePath, "RGB")

    #DWT on cover_img
    cover_red_dwt_subbands, cover_green_dwt_subbands, cover_blue_dwt_subbands = dwt_rgb_image(cover_img)

    #SVD on cover_img
    U_cover_red_channel, S_cover_red_channel, V_cover_red_channel = svd(cover_red_dwt_subbands[2])
    U_cover_green_channel, S_cover_green_channel, V_cover_green_channel = svd(cover_green_dwt_subbands[2])
    U_cover_blue_channel, S_cover_blue_channel, V_cover_blue_channel = svd(cover_blue_dwt_subbands[2])


    #DWT on watermark
    watermark_red_dwt_subbands, watermark_green_dwt_subbands, watermark_blue_dwt_subbands = dwt_rgb_image(watermark_img)

    #SVD on watermark
    U_watermark_red, S_watermark_red, V_watermark_red = svd(watermark_red_dwt_subbands[2])
    U_watermark_green, S_watermark_green, V_watermark_green = svd(
        watermark_green_dwt_subbands[2])
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd(
        watermark_blue_dwt_subbands[2])

    # Embed watermark
    S_watermarked_red = embed_watermark_dwt_svd(S_cover_red_channel,S_watermark_red)
    S_watermarked_green = embed_watermark_dwt_svd(S_cover_green_channel,S_watermark_green)
    S_watermarked_blue = embed_watermark_dwt_svd(S_cover_blue_channel,S_watermark_blue)

    # replace

    # reverse svd
    watermarked_img_red = reverse_svd( U_cover_red_channel, S_watermarked_red, V_cover_red_channel )
    watermarked_img_green = reverse_svd(U_cover_green_channel, S_watermarked_green, V_cover_green_channel)
    watermarked_img_blue = reverse_svd(U_cover_blue_channel, S_watermarked_blue, V_cover_green_channel)

    # replace cover_img_HL subband with watermarked HL subband
    cover_red_dwt_subbands[2] = watermarked_img_red
    cover_green_dwt_subbands[2] = watermarked_img_green
    cover_blue_dwt_subbands[2] = watermarked_img_blue

    # reverse dwt on watermarked cover image
    red_channel, green_channel, blue_channel = reverse_dwt_rgb(cover_red_dwt_subbands, cover_green_dwt_subbands,
                                                               cover_blue_dwt_subbands)

    # watermarked_img = np.dstack((b, g, r))
    watermarked_img = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'watermarked_image_DWT_SVD_RGB_HL.jpg'
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_SVD_RGB_HL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    cover_img = readFile(coverImagePath, "RGB")
    watermark_img = readFile(watermarkImagePath, "RGB")

    # DWT on cover_img
    cover_red_dwt_subbands, cover_green_dwt_subbands, cover_blue_dwt_subbands = dwt_rgb_image(cover_img)

    # SVD on cover_img
    U_cover_red_channel, S_cover_red_channel, V_cover_red_channel = svd(cover_red_dwt_subbands[2])
    U_cover_green_channel, S_cover_green_channel, V_cover_green_channel = svd(cover_green_dwt_subbands[2])
    U_cover_blue_channel, S_cover_blue_channel, V_cover_blue_channel = svd(cover_blue_dwt_subbands[2])

    # DWT on watermark
    watermark_red_dwt_subbands, watermark_green_dwt_subbands, watermark_blue_dwt_subbands = dwt_rgb_image(watermark_img)

    # SVD on watermark
    U_watermark_red, S_watermark_red, V_watermark_red = svd(watermark_red_dwt_subbands[2])
    U_watermark_green, S_watermark_green, V_watermark_green = svd(
        watermark_green_dwt_subbands[2])
    U_watermark_blue, S_watermark_blue, V_watermark_blue = svd(
        watermark_blue_dwt_subbands[2])

    # Extracting embeded watermark

    watermarked_red_dwt_subbands, watermarked_green_dwt_subbands, watermarked_blue_dwt_subbands = dwt_rgb_image(
        watermarked_img)
    # svd on watermarked LL subband image
    U_watermarked_red_channel, S_watermarked_red_channel, V_watermarked_red_channel = svd(
        watermarked_red_dwt_subbands[2])
    U_watermarked_green_channel, S_watermarked_green_channel, V_watermarked_green_channel = svd(
        watermarked_green_dwt_subbands[2])
    U_watermarked_blue_channel, S_watermarked_blue_channel, V_watermarked_blue_channel = svd(
        watermarked_blue_dwt_subbands[2])

    # Extracting watermark at HL subband
    S_ewatr = extract_watermark_dwt_svd(S_cover_red_channel, S_watermarked_red_channel)
    S_ewatg = extract_watermark_dwt_svd(S_cover_green_channel, S_watermarked_green_channel)
    S_ewatb = extract_watermark_dwt_svd(S_cover_blue_channel, S_watermarked_blue_channel)

    # reverse SVD on watermarked image, return subband HL
    extracted_watermark_red_HL = reverse_svd(U_watermark_red, S_ewatr, V_watermark_red)
    extracted_watermark_green_HL = reverse_svd(U_watermark_green, S_ewatg, V_watermark_green)
    extracted_watermark_blue_HL = reverse_svd(U_watermark_blue, S_ewatb, V_watermark_blue)

    # replace watermark subband with watermarked LL subbandr
    watermark_red_dwt_subbands[2] = extracted_watermark_red_HL
    watermark_green_dwt_subbands[2] = extracted_watermark_green_HL
    watermark_blue_dwt_subbands[2] = extracted_watermark_blue_HL

    red_channel, green_channel, blue_channel = reverse_dwt_rgb(watermark_red_dwt_subbands, watermark_green_dwt_subbands,
                                                               watermark_blue_dwt_subbands)

    # compose color channels to BGR image
    extracted_watermark = combine_rgb_channels_to_bgr_img(red_channel, green_channel, blue_channel)

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_SVD_RGB_HL.jpg'
    cv2.imwrite(out_path, extracted_watermark);
    print(out_path)
    return out_path


def DWT_SVD_GRAY_LL_EMBED(coverImagePath, watermarkImagePath):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")

    # dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_LL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    # dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_LL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()

    S_wimg = S_c_img + (0.01 * S_w_img)
    wimgr = np.dot(U_c_img, np.dot(S_wimg, V_c_img.transpose()))
    watermarked_img = pywt.idwt2((wimgr, (c_LH, c_HL, c_HH)), 'haar')

    cv2.imwrite(IMAGES_DIR + 'watermarked_image_DWT_SVD_GRAY_LL.jpg', watermarked_img)

    return watermarked_img


def DWT_SVD_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")

    # dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_LL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    # dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_LL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_LL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()
    # Extracting embeded watermark

    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    U_wed_img, S_wed_img, V_wed_img = np.linalg.svd(wed_LL, full_matrices=1, compute_uv=1)

    S = np.zeros((np.shape(wed_LL)))
    np.fill_diagonal(S, S_wed_img)
    S_wed_img = S
    V_wed_img = V_wed_img.T.conj()

    S_ewat = (S_wed_img - S_c_img) / 0.01
    ewatr = np.dot(U_w_img, np.dot(S_ewat, V_w_img.transpose()))

    extracted_watermark = pywt.idwt2((ewatr, (w_LH, w_HL, w_HH)), 'haar')

    cv2.imwrite(IMAGES_DIR + 'extracted_watermark_DWT_SVD_GRAY_LL.jpg', extracted_watermark);
    return extracted_watermark


def DWT_SVD_GRAY_HL_EMBED(coverImagePath, watermarkImagePath):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")

    # dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_HL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    # dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_HL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()

    S_wimg = S_c_img + (0.01 * S_w_img)
    wimgr = np.dot(U_c_img, np.dot(S_wimg, V_c_img.transpose()))
    watermarked_img = pywt.idwt2((c_LL, (c_LH, wimgr, c_HH)), 'haar')

    out_path = IMAGES_DIR + 'watermarked_image_DWT_SVD_GRAY_HL.jpg', watermarked_img
    cv2.imwrite(out_path, watermarked_img)

    return out_path


def DWT_SVD_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath, watermarked_img):
    coverImage = readFile(coverImagePath, "GRAY")
    watermarkImage = readFile(watermarkImagePath, "GRAY")

    # dwt on cover image
    c_LL, (c_LH, c_HL, c_HH) = pywt.dwt2(coverImage, 'haar')
    # svd on cover image LL
    U_c_img, S_c_img, V_c_img = np.linalg.svd(c_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(c_HL)))
    np.fill_diagonal(S, S_c_img)
    S_c_img = S
    V_c_img = V_c_img.T.conj()

    # dwt on watermark image
    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')
    # svd on watermark image LL
    U_w_img, S_w_img, V_w_img = np.linalg.svd(w_HL, full_matrices=1, compute_uv=1)
    S = np.zeros((np.shape(w_HL)))
    np.fill_diagonal(S, S_w_img)
    S_w_img = S
    V_w_img = V_w_img.T.conj()

    # Extracting embeded watermark

    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    U_wed_img, S_wed_img, V_wed_img = np.linalg.svd(wed_HL, full_matrices=1, compute_uv=1)

    S = np.zeros((np.shape(wed_HL)))
    np.fill_diagonal(S, S_wed_img)
    S_wed_img = S
    V_wed_img = V_wed_img.T.conj()

    S_ewat = (S_wed_img - S_c_img) / 0.01
    ewatr = np.dot(U_w_img, np.dot(S_ewat, V_w_img.transpose()))

    extracted_watermark = pywt.idwt2((w_LL, (w_LH, ewatr, w_HH)), 'haar')

    out_path = IMAGES_DIR + 'extracted_watermark_DWT_SVD_GRAY_HL.jpg', extracted_watermark
    cv2.imwrite(out_path);
    return out_path


def readFile(path, colourType):  # colour type == GRAY or RGB
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

    watermarked_img_rgb_hl = DWT_SVD_RGB_HL_EMBED(coverImagePath, watermarkImagePath)
    extracted_img_rgb_hl = DWT_SVD_RGB_HL_EXTRACT(coverImagePath, watermarkImagePath, readFile(watermarked_img_rgb_hl, "RGB"))

    # watermarked_img_rbg_ll = DWT_SVD_RGB_LL_EMBED(coverImagePath, watermarkImagePath)
    # extracted_img_rbg_ll = DWT_SVD_RGB_LL_EXTRACT(coverImagePath, watermarkImagePath,
    #                                               readFile(watermarked_img_rbg_ll, "RGB"))

    # watermarked_img_gray_hl = DWT_SVD_GRAY_HL_EMBED(coverImagePath, watermarkImagePath)
    # extracted_img_gray_hl = DWT_SVD_GRAY_HL_EXTRACT(coverImagePath, watermarkImagePath, readFile(watermarked_img_gray_hl, "RGB"))
    #
    # watermarked_img_gray_ll = DWT_SVD_GRAY_LL_EMBED(coverImagePath, watermarkImagePath)
    # extracted_img_gray_ll = DWT_SVD_GRAY_LL_EXTRACT(coverImagePath, watermarkImagePath, readFile(watermarked_img_gray_ll, "RGB"))
