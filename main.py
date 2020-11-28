import cv2
import numpy as np

import SVD as SVD
import DWT_SVD as DWT_SVD
import DWT as DWT
# import DWT_DCT as DWT_DCT

from attacks import Attacks as Attacks


def compression(quality):
    return Attacks.compression("lenna_256.jpg", quality)

def grayMenu():
    coverImage = cv2.imread('images\\mandrill_512.jpg', 0)
    cv2.imshow('orginal image', coverImage)
    watermarkImage = cv2.imread('images\\lenna_512.jpg', 0)
    cv2.imshow('watermark', watermarkImage)

    transformationOptions = {1: DWT_SVD.DWT_SVD_GRAY_LL, 2: DWT_SVD.DWT_SVD_GRAY_HL}
    transformationVal = int(input(
        '\n\033[92mWhat type of embedding you want to perform?\033[0m \n1.SVD-DWT_GRAY_LL \n2.SVD-DWT_GRAY_HL'))
    transformationOptions[transformationVal](coverImage,watermarkImage)

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack = int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks'))

    if processOptions[processOrAttack] == "watermark":
        transformationOptions[transformationVal](coverImage,watermarkImage)
    elif processOptions[processOrAttack] == "attack":
        transformationName = ""
        if (transformationVal == 1):
            transformationName = "DWT_SVD_GRAY_LL.jpg"
        elif(transformationVal == 2):
            transformationName = "DWT_SVD_GRAY_HL.jpg"
        elif (transformationVal == 3):
            transformationName = "DCT-DWT_GRAY_LL.jpg"
        elif (transformationVal == 4):
            transformationName = "DWT_GRAY_LL.jpg"
        elif (transformationVal == 5):
            transformationName = "DCT_GRAY_LL.jpg"
        attacksMenu(transformationName)




def rgbMenu():
    coverImage = cv2.imread('images\\mandrill_512.jpg', 8)
    cv2.imshow('orginal image', coverImage)
    watermarkImage = cv2.imread('images\\lenna_512.jpg', 8)
    cv2.imshow('watermark image', watermarkImage)

    transformationOptions = {1: DWT_SVD.DWT_SVD_RGB_LL, 2: DWT_SVD.DWT_SVD_RGB_HL}
    transformationVal = int(input(
        '\n\033[92mWhat type of transformation do you want to perform?\033[0m \n1.SVD-DWT_RGB_LL \n2.SVD-DWT_RGB_HL \n3.DCT-DWT_RGB_LL \n4.DWT_RGB_LL \n5.DCT_RGB_LL '))

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack= int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks'))

    if processOptions[processOrAttack] == "watermark":
        transformationOptions[transformationVal](coverImage,watermarkImage)
    elif processOptions[processOrAttack] == "attack":
        transformationName = ""
        if (transformationVal == 1):
            transformationName = "DWT_SVD_RGB_LL.jpg"
        elif(transformationVal == 2):
            transformationName = "DWT_SVD_RGB_HL.jpg"
        elif (transformationVal == 3):
            transformationName = "DCT-DWT_RGB_LL.jpg"
        elif (transformationVal == 4):
            transformationName = "DWT_RGB_LL.jpg"
        elif (transformationVal == 5):
            transformationName = "DCT_RGB_LL.jpg"
        attacksMenu(transformationName)

def attacksMenu(transformationName):
    extractedWatermarkPath = "extracted_watermark_" + transformationName
    watermarkPath = "watermarked_image_" + transformationName

    extractedWatermark = cv2.imread(extractedWatermarkPath, 8)
    cv2.imshow('extracted watermark image test from attacks', extractedWatermark)

    watermark = cv2.imread(watermarkPath, 8)
    cv2.imshow('watermark image test from attacks', watermark)

    options = {1: Attacks.compression, 2: Attacks.distorition, 3: Attacks.transformation}
    val = int(input(
        '\n\033[92mWhat kind of attack to perform?\033[0m \n1.Compression \n2.Distortion \n3.Transformation'))
    # options[val]()
    print("" + str(options[val]))

if __name__ == "__main__":

    options = {1: rgbMenu, 2: grayMenu}
    val = int(input('\033[92mDo you want to process RGB or GRAY images? \033[0m \n1.RGB \n2.GRAY'))
    options[val]()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# if __name__ == "__main__":
#     coverImage = cv2.imread('mandrill.jpg', 0)
#     watermarkImage = cv2.imread('lenna_256.jpg', 0)
#
#     options = {1: DWT,
#                2: SVD,
#                3: DWT_SVD,
#                4: DWT_DCT,
#                5: compression
#                }
#
#     choice = int(input('What type of embedding you want to perform?\n1.DWT\n2.SVD\n3.SVD-DWT\n4.Compression\n'))
#     if choice == 1:
#         opt = int(input('What type of embedding you want to perform?\n1.DWT_GRAY_EMBED\n2.DWT_RGB_LL\n'))
#         if opt == 1:
#             DWT.DWT_RGB_LL("mandrill.jpg", "lenna.jpg")
#         elif opt == 2:
#             DWT.DWT_GRAY_LL("mandrill.jpg", "lenna.jpg")
#     elif choice > 1 and choice < 4:
#         options[choice](coverImage, watermarkImage)
#     elif choice == 5:
#         quality = int(input('Compression quality <0, 100>\n'))
#         options[choice](quality)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
