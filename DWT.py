import cv2
import numpy as np
import pywt
import attacks.Attacks as attacks
import logging
import PIL
logging.basicConfig(level=logging.DEBUG)

IMAGES_DIR = "processed_images\\"

def DWT_RGB_LL(coverImagePath, watermarkImagePath, attaack_name = None, *args):
    coverImage = read_file(coverImagePath, "RGB")
    watermarkImage = read_file(watermarkImagePath, "RGB")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # get color cover chanels BGR
    cover_red1 = coverImage[:, :, 2]
    cover_green1 = coverImage[:, :, 1]
    cover_blue1 = coverImage[:, :, 0]

    # dwt on cover image on particular color channels
    cr_LL, (cr_LH, cr_HL, cr_HH) = pywt.dwt2(cover_red1, 'haar')
    cg_LL, (cg_LH, cg_HL, cg_HH) = pywt.dwt2(cover_green1, 'haar')
    cb_LL, (cb_LH, cb_HL, cb_HH) = pywt.dwt2(cover_blue1, 'haar')

    # get color w   atermark chanels BGR
    watermark_red = watermarkImage[:, :, 2]
    watermark_green = watermarkImage[:, :, 1]
    watermark_blue = watermarkImage[:, :, 0]

    # dwt on watermark on particular color channels
    wr_LL, (wr_LH, wr_HL, wr_HH) = pywt.dwt2(watermark_red, 'haar')
    wg_LL, (wg_LH, wg_HL, wg_HH) = pywt.dwt2(watermark_green, 'haar')
    wb_LL, (wb_LH, wb_HL, wb_HH) = pywt.dwt2(watermark_blue, 'haar')

    # Embedding
    red_LL_embeded = cr_LL + 0.01 * wr_LL
    green_LL_embeded = cg_LL + 0.01 * wg_LL
    blue_LL_embeded = cb_LL + 0.01 * wb_LL

    r = pywt.idwt2((red_LL_embeded, (cr_LH, cr_HL, cr_HH)), 'haar')
    g = pywt.idwt2((green_LL_embeded, (cg_LH, cg_HL, cg_HH)), 'haar')
    b = pywt.idwt2((blue_LL_embeded, (cb_LH, cb_HL, cb_HH)), 'haar')
    watermarked_img = np.dstack((b, g, r))


    if attaack_name is None:
        cv2.imshow('Watermarked Image', watermarked_img.astype(np.uint8))
        cv2.imwrite(''+IMAGES_DIR +'watermarked_image_DWT_RGB_LL.jpg', watermarked_img)

    # Attacks##
    else:
        watermarked_img = attacks.perform_attack(watermarked_img, "DWT_RGB_LL", attaack_name, *args)
        cv2.imshow('Watermarked Image after' + attaack_name + ' attack', watermarked_img.astype(np.uint8))
        cv2.imwrite('watermarked_image_DWT_RGB_LL_' + attaack_name + '.jpg', watermarked_img)
    print("xD")
    # # Extraction
    ### fir size of rwed_LL param
    if (attaack_name == "resize_attack"):

        # watermarked_img.resize((512, 512))
        watermarked_img = cv2.resize(watermarked_img, (512, 512))
        print("xD")
    ###
    print ("xD")
    cover_red1 = watermarked_img[:, :, 2]
    cover_green1 = watermarked_img[:, :, 1]
    cover_blue1 = watermarked_img[:, :, 0]

    # dwt on watermarked image color channels
    rwed_LL, (rwed_LH, rwed_HL, rwed_HH) = pywt.dwt2(cover_red1, 'haar')
    gwed_LL, (gwed_LH, gwed_HL, gwed_HH) = pywt.dwt2(cover_green1, 'haar')
    bwed_LL, (bwed_LH, bwed_HL, bwed_HH) = pywt.dwt2(cover_blue1, 'haar')

    #Extract algorithm on color channels

    extracted_watermark_LL_r = (rwed_LL - cr_LL) / 0.01
    extracted_watermark_LL_g = (gwed_LL - cg_LL) / 0.01
    extracted_watermark_LL_b = (bwed_LL - cb_LL) / 0.01

    # idwt on color channels - getting waermark image
    rw = pywt.idwt2((extracted_watermark_LL_r, (wr_LH, wr_HL, wr_HH)), 'haar')
    gw = pywt.idwt2((extracted_watermark_LL_g, (wg_LH, wg_HL, wg_HH)), 'haar')
    bw = pywt.idwt2((extracted_watermark_LL_b, (wb_LH, wb_HL, wb_HH)), 'haar')

    extracted_watermark = np.dstack((bw, gw, rw))

    if attaack_name is None:
        cv2.imshow('Extracted watermark', extracted_watermark.astype(np.uint8))
        cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_RBG_LL.jpg', extracted_watermark)
    else:
        cv2.imshow('Extracted watermark after ' + attaack_name + ' attack', extracted_watermark.astype(np.uint8))
        cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_RBG_LL_' + attaack_name + '.jpg', extracted_watermark)

def DWT_GRAY_LL(coverImagePath, watermarkImagePath, attack_name = None, *args):
    coverImage = read_file(coverImagePath, "GRAY")
    watermarkImage = read_file(watermarkImagePath, "GRAY")
    cv2.imshow('orginal image', coverImage)
    cv2.imshow('watermark image', watermarkImage)

    # DWT on cover image

    coeffC = pywt.dwt2(coverImage, 'haar')
    cr_LL, (cr_LH, cr_HL, cr_HH) = coeffC

    # DWT on watermark image

    w_LL, (w_LH, w_HL, w_HH) = pywt.dwt2(watermarkImage, 'haar')

    # Embedding
    coeffW = (cr_LL + 0.1 * w_LL, (cr_LH, cr_HL, cr_HH))
    watermarked_img = pywt.idwt2(coeffW, 'haar')
    # cv2.imshow('Watermarked Image', watermarkedImage.astype(np.uint8))
    # cv2.imwrite('' + IMAGES_DIR +'watermarked_image_DWT_GRAY_LL.jpg', watermarkedImage)
 # attacks
    if attack_name is None:
        logging.debug("attack name is None, Embed watermark. Method DWT_GRAY_LL")
        cv2.imshow('Watermarked Image', watermarked_img)
        cv2.imwrite('' + IMAGES_DIR + 'Watermarked Image_DWT_GRAY_LL.jpg', watermarked_img)

    else:
        watermarked_img = attacks.perform_attack(watermarked_img, "DWT_GRAY_LL", attack_name, *args)
        logging.debug("attack name is: " + attack_name + ",Embed watermark DWT_GRAY_LL")
        cv2.imshow('Watermarked Image after ' + attack_name + ' attack', watermarked_img.astype(np.uint8))
        cv2.imwrite('' + IMAGES_DIR + 'Watermarked Image_DWT_RBG_LL' + attack_name + '.jpg', watermarked_img)
        logging.debug("attack name is: " + attack_name + ", Embedding IS DONE")
    ### fir size of rwed_LL param
    if (attack_name == "resize_attack"):
        watermarked_img = cv2.resize(watermarked_img, (512, 512))
    ###

    # Extraction
    wed_LL, (wed_LH, wed_HL, wed_HH) = pywt.dwt2(watermarked_img, 'haar')
    extracted = (wed_LL - cr_LL) / 0.1   #extracted = (hA - 0.4 * cA) / 0.1
    # extracted *= 255
    extracted_watermark = pywt.idwt2((extracted, (w_LH, w_HL, w_HH)), 'haar')
    extracted_watermark = np.uint8(extracted_watermark)
    if attack_name is None:
        logging.debug("attack name is None. Method DWT_GRAY_LL")
        cv2.imshow('Extracted', extracted_watermark)
        cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_GRAY_LL.jpg', extracted_watermark)

    else:
      #  watermarked_img = attacks.perform_attack(watermarked_img, attaack_name, args)
        logging.debug("attack name is: " + attack_name + ", DWT_GRAY_LL")
        cv2.imshow('Extracted watermark after s' + attack_name + ' attack', extracted_watermark.astype(np.uint8))
        cv2.imwrite('' + IMAGES_DIR + 'extracted_watermark_DWT_RBG_LL' + attack_name + '.jpg', extracted_watermark)
        logging.debug("attack name is: " + attack_name + ", IS DONE")

def read_file(path, color): # color == GRAY or RGB
    if color == "GRAY":
        img = cv2.imread(path, 0)
        return img
    elif color == "RGB":
        img = cv2.imread(path, 8)
        return img
    else:
        print("failed to read image")


#TESTS


if __name__ == "__main__":
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath= 'images\\lenna_512.jpg'
    watermarkedImagePath_DWT_GRAY_LL = 'watermarked_image_DWT_GRAY_LL.bmp'
    watermarkedImagePath_DWT_RGB_LL = 'watermarked_image_DWT_RGB_LL.bmp'
    options = {1: DWT_RGB_LL,2: DWT_GRAY_LL, 0: exit}
    val = int(input('\033[92mRun Choose operatin \033[0m \n[1] DWT_RGB_LL_rotate_image \n[2] DWT_RGB_LL_distortion '
                    + '\n[3] DWT_RGB_LL_resize_attack \n[4] DWT_RGB_LL_compression \n[5] DWT_RGB_LL_gussian_noise '
                    + '\n[6] DWT_RGB_LL_salt_and_pepper \n[7] DWT_RGB_LL normal run '
                    +'\n[8] DWT_GRAY_LL_rotate_image \n[9] DWT_GRAY_LL_distortion '
                    + '\n[10] DWT_GRAY_LL_resize_attack \n[11] DWT_GRAY_LL_compression \n[12] DWT_GRAY_LL_gussian_noise '
                    + '\n[13] DWT_GRAY_LL_salt_and_pepper \n[14] DWT_GRAY_LL normal run '
                    +'\n[0] EXIT\n'))
    if val == 1:
        options[1](coverImagePath, watermarkImagePath, "rotate_image",90)
    elif val == 2:
        options[1](coverImagePath, watermarkImagePath, "distortion")
    elif val == 3:
        options[1](coverImagePath, watermarkImagePath, "resize_attack", 200)
        #Image.open
    elif val == 4:
        options[1](coverImagePath, watermarkImagePath, "compression", 1)
    elif val == 5:
        options[1](coverImagePath, watermarkImagePath, "gaussian_noise")
    elif val == 6:
        options[1](coverImagePath, watermarkImagePath,"salt_and_pepper", 0.1, 1)
    elif val == 7:
        options[1](coverImagePath, watermarkImagePath)

    if val == 8:
        options[2](coverImagePath, watermarkImagePath, "rotate_image",90)
    elif val == 9:
        options[2](coverImagePath, watermarkImagePath, "distortion")
    elif val == 10:
        options[2](coverImagePath, watermarkImagePath, "resize_attack", 200)
        #Image.open
    elif val == 11:
        options[2](coverImagePath, watermarkImagePath, "compression", 1)
    elif val == 12:
        options[2](coverImagePath, watermarkImagePath, "gaussian_noise")
    elif val == 13:
        options[2](coverImagePath, watermarkImagePath, "salt_and_pepper", 0.1, 1)
    elif val == 14:
        options[2](coverImagePath, watermarkImagePath)
    else:
        options[0]()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# add to Attacks
# def perform_attack(watermarked_img, method, *args):
#     switcher = {
#         "rotate_image": attacks.rotate_image(watermarked_img, args[0]),
#         "distortion": attacks.distorition(watermarked_img),
#         "resize_attack": attacks.resize_attack(watermarked_img, args[0]),
#         "compression": attacks.compression(watermarked_img, args[0]),
#         "gaussian_noise": attacks.gaussian_noise(watermarked_img),
#         "salt_and_pepper": attacks.salt_and_pepper(watermarked_img, args[0],args[1]),
#     }
#     imageAfterAttack = switcher[method]
#     return imageAfterAttack

