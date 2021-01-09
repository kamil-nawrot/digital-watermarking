import sys

import cv2

import Attacks as Attacks
import DWT as DWT
import DWT_DCT as DWT_DCT
import DWT_SVD as DWT_SVD


def compression(quality):
    return Attacks.compression("lenna_256.jpg", quality)

def grayMenu():
    coverImagePathDwtSvd='images\\mandrill_512.jpg'
    watermarkImagePathDwtSvd= 'images\\lenna_512.jpg'
    coverImagePathDctDwt = 'images/lenna_256.jpg'
    watermarkImagePathDctDwt = 'images/mandrill_256.jpg'

    transformationOptions = {1: DWT_SVD.DWT_SVD_GRAY_LL, 2: DWT_SVD.DWT_SVD_GRAY_HL, 5: DWT.DWT_GRAY_LL}
    transformationVal = int(input(
        '\n\033[92mWhat type of embedding you want to perform?\033[0m \n1.DWT-SVD_LL \n2.DWT-SVD_HL \n3.DCT-DWT_LL \n4.DCT-DWT_HL \n5.DWT_LL\n'))

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack = int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks\n'))

    if processOptions[processOrAttack] == "watermark":
        if transformationVal != 3 and transformationVal != 4:
            transformationOptions[transformationVal](coverImagePathDwtSvd,watermarkImagePathDwtSvd)
        elif transformationVal == 3:
            baseImage = DWT_DCT.DWTDCT("base", coverImagePathDctDwt, (1024, 1024), 0)
            watermarkImage = DWT_DCT.DWTDCT("watermark", watermarkImagePathDctDwt, (128, 128), 0)
            baseImage.embed_watermark('LL', watermarkImage)
            baseImage.display()
            baseImage.save('watermarked_image_DCT_DWT_GRAY_LL.jpg')
            reconstructedImage = DWT_DCT.DWTDCT("watermarked", "watermarked_image_DCT_DWT_GRAY_LL.jpg")
            reconstructedImage.extract_watermark('LL', 128)
        elif transformationVal == 4:
            baseImage = DWT_DCT.DWTDCT("base", "images/lenna_256.jpg", (1024, 1024), 0)
            watermarkImage = DWT_DCT.DWTDCT("watermark", "images/mandrill_256.jpg", (128, 128), 0)
            baseImage.embed_watermark('HL', watermarkImage)
            baseImage.display()
            baseImage.save('watermarked_image_DCT_DWT_GRAY_HL.jpg')
            reconstructedImage = DWT_DCT.DWTDCT("watermarked", "watermarked_image_DCT_DWT_GRAY_HL.jpg")
            reconstructedImage.extract_watermark('HL', 128)


    elif processOptions[processOrAttack] == "attack":
        transformationName = getTransformName(transformationVal, "GRAY")
        attacksMenu(transformationName)




def rgbMenu():
    coverImagePathDwtSvd = 'images\\mandrill_512.jpg'
    watermarkImagePathDwtSvd = 'images\\lenna_512.jpg'
    coverImagePathDctDwt = 'images/lenna_256.jpg'
    watermarkImagePathDctDwt = 'images/mandrill_256.jpg'


    transformationOptions = {1: DWT_SVD.DWT_SVD_RGB_LL, 2: DWT_SVD.DWT_SVD_RGB_HL, 5: DWT.DWT_RGB_LL}
    transformationVal = int(input(
        '\n\033[92mWhat type of transformation do you want to perform?\033[0m \n1.DWT_SVD_LL \n2.DWT_SVD_HL \n3.DCT-DWT_LL\n4.DCT_DWT_HL \n5.DWT_LL\n'))

    processOptions = {1: "watermark", 2: "attack"}
    processOrAttack= int(input('\n\033[92mwhat do you want to perform?\033[0m \n1.Perform watermarking \n2.Perform attacks\n '))

    if processOptions[processOrAttack] == "watermark":
        if transformationVal != 3 and transformationVal !=4:
            # DWT param
            transformationOptions[transformationVal](coverImagePathDwtSvd,watermarkImagePathDwtSvd,"")
        elif transformationVal == 3:
            baseImage = DWT_DCT.DWTDCT("base", coverImagePathDctDwt, (1024, 1024), 8)
            watermarkImage = DWT_DCT.DWTDCT("watermark", watermarkImagePathDctDwt, (128, 128), 8)
            baseImage.embed_watermark('LL', watermarkImage)
            baseImage.display()
            baseImage.save('watermarked_image_DCT_DWT_RGB_LL.jpg')
            reconstructedImage = DWT_DCT.DWTDCT("watermarked", "watermarked_image_DCT_DWT_RGB_LL.jpg")
            reconstructedImage.extract_watermark('LL', 128)
        elif transformationVal == 4:
            baseImage = DWT_DCT.DWTDCT("base", "images/lenna_256.jpg", (1024, 1024), 8)
            watermarkImage = DWT_DCT.DWTDCT("watermark", "images/mandrill_256.jpg", (128, 128), 8)
            baseImage.embed_watermark('HL', watermarkImage)
            baseImage.display()
            baseImage.save('watermarked_image_DCT_DWT_RGB_HL.jpg')
            reconstructedImage = DWT_DCT.DWTDCT("watermarked", "watermarked_image_DCT_DWT_RGB_HL.jpg")
            reconstructedImage.extract_watermark('HL', 128)
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
        '\n\033[92mWhat kind of attack to perform?\033[0m \n1.Compression \n2.Distortion \n3.Transformation\n'))
    # options[val]()
    print("" + str(options[val]))

def exit():
    sys.exit()



if __name__ == "__main__":


    options = {1: rgbMenu, 2: grayMenu, 0: exit}
    val = int(input('\033[92mDo you want to process RGB or GRAY images? \033[0m \n1.RGB \n2.GRAY \n0.EXIT\n'))
    options[val]()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



