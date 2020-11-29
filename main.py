import cv2
import numpy as np

import SVD as SVD
import DWT_SVD as DWT_SVD
import DWT as DWT
import DWT_DCT as DWT_DCT

from attacks import Attacks as Attacks


def compression(quality):
    return Attacks.compression("lenna_256.jpg", quality)

def grayMenu():
    coverImagePath='images\\mandrill_512.jpg'
    watermarkImagePath= 'images\\lenna_512.jpg'

    transformationOptions = {1: DWT_SVD.DWT_SVD_GRAY_LL, 2: DWT_SVD.DWT_SVD_GRAY_HL, 4: DWT.DWT_GRAY_LL}
    transformationVal = int(input(
        '\n\033[92mWhat type of embedding you want to perform?\033[0m \n1.SVD-DWT_LL \n2.SVD-DWT_HL \n4.DWT_LL'))
    # transformationOptions[transformationVal](coverImagePath,watermarkImagePath)

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack = int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks'))

    if processOptions[processOrAttack] == "watermark":
        transformationOptions[transformationVal](coverImagePath,watermarkImagePath)
    elif processOptions[processOrAttack] == "attack":
        transformationName = getTransformName(transformationVal, "GRAY")
        attacksMenu(transformationName)




def rgbMenu():
    coverImagePath = 'images\\mandrill_512.jpg'
    watermarkImagePath = 'images\\lenna_512.jpg'

    transformationOptions = {1: DWT_SVD.DWT_SVD_RGB_LL, 2: DWT_SVD.DWT_SVD_RGB_HL, 4: DWT.DWT_RGB_LL}
    transformationVal = int(input(
        '\n\033[92mWhat type of transformation do you want to perform?\033[0m \n1.SVD-DWT_LL \n2.SVD-DWT_HL \n3.DCT-DWT_LL \n4.DWT_LL'))

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack= int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks'))

    if processOptions[processOrAttack] == "watermark":
        if transformationVal != 3:
            transformationOptions[transformationVal](coverImagePath,watermarkImagePath)
        else:
            baseImage = DWT_DCT.Image("base", "images/lenna_256.jpg", (1024, 1024))
            watermarkImage = DWT_DCT.Image("watermark", "images/mandrill_256.jpg", (128, 128))
            baseImage.embed_watermark('LL', watermarkImage)
            baseImage.display()
            baseImage.save('watermarked_image.jpg')
            reconstructedImage = DWT_DCT.Image("watermarked", "watermarked_image.jpg")
            reconstructedImage.extract_watermark('LL', 128)
    elif processOptions[processOrAttack] == "attack":
        transformationName = getTransformName(transformationVal, "RGB")
        attacksMenu(transformationName)

def getTransformName(transformationVal, colourOption):
    transformationName = ""
    if (transformationVal == 1):
        transformationName = "DWT_SVD_" + colourOption + "_LL.jpg"
    elif (transformationVal == 2):
        transformationName = "DWT_SVD_" + colourOption + "_HL.jpg"
    elif (transformationVal == 3):
        transformationName = "DCT-DWT_" + colourOption + "_LL.jpg"
    elif (transformationVal == 4):
        transformationName = "DWT_" + colourOption + "_LL.jpg"
    elif (transformationVal == 5):
        transformationName = "DCT_" + colourOption + "_LL.jpg"
    return transformationName

def attacksMenu(transformationName):
    extractedWatermarkPath = "extracted_watermark_" + transformationName
    watermarkPath = "watermarked_image_" + transformationName

    extractedWatermark = cv2.imread(extractedWatermarkPath, 8)
    # cv2.imshow('extracted watermark image test from attacks', extractedWatermark)

    watermark = cv2.imread(watermarkPath, 8)
    # cv2.imshow('watermark image test from attacks', watermark)

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
