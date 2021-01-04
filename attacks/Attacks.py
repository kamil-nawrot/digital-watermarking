import calendar
import time

import cv2
from PIL import Image


def checkPSNR(original, attacked):
    # img1 = cv2.imread("images/" + original)
    img1 = cv2.imread("../images/lenna_256.jpg")
    # img2 = cv2.imread("images/" + attacked)
    img2 = cv2.imread("compressed/Compressed_1606579782_lenna_256.jpg")
    psrn = cv2.PSNR(img1, img2)
    print(psrn)
    return psrn


def compression(filename, quality):
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    im = Image.open("images/" + filename)
    ts = calendar.timegm(time.gmtime())
    im.save("attacks/compressed/Compressed_" + str(quality) + "_" + str(ts) + "_" + filename, optimize=True, quality=quality)
    return ''


def distorition(filename):
    # https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
    # https://www.geeksforgeeks.org/python-distort-method-in-wand/
    return ''


def transformation(filename):
    # https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
    im = Image.open(filename)
    return ''


checkPSNR("", "")

# KAMIL PSEUDO CODE/ SOME KIND OF CODE


def performAttacks():
    #arrayWithExtractedWatermarks
    #Attacks for DWT_GRAY_LL
    #watermarkedImage = DWT.DWT_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
    #watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
    #DWT_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

    #Attack for DWT_SVD_GRAY_LL
    # watermarkedImage = DWT.DWT_SVD_GRAY_LL_EMBED(coverImagePath,watermarkImagePath)
    # watermarkedImagesAfterAttack = performAttacksOnWatermarkedImage(watermarkedImage)
    # DWT_SVD_GRAY_LL_watermarksAfterAttack = extractWatermarkFromArray(watermarkedImagesAfterAttack)

def performAttacksOnWatermarkedImage(imageWithWatermark,method): #method = DWT.DWT_GRAY_LL

    #saltAndPepperAttackResult = saltAndPepper(imageWithWatermark)
    #cv2.imwrite(''+method+'_watermarkedImageAfterSaltAndPepperAttack', saltAndPepperAttackResult) #zapis
    #GausianAttackResult = gausian(imageWithWatermark)
    #cv2.imwrite
    #transformationAttackResult = transformation(imageWithWatermark)
    #cv2.imwrite
    #compressionAttackResult = compression(imageWithWatermark)
    #cv2.imwrite

    #attacksTable = [pathTosaltAndPepperAttack,pathToGausianAttack, pathToTransformationAttack, pathToCompressionAttack ]

    #return attacksTable

def extractWatermarksFromArray(tableWithAttackedImagesPaths,method): # method = DWT.DWT_GRAY_LL
    #counter = 0
    #method + "_EXTRACT"
    #extractedWatermarksArray

    #for path in tableWithAttackedImagesPaths
        #extractedWatermark = runAppropriateExtracMethod(path, method)
        #extractedWatermarksArray[counter] = extractedWatermark
        #counter ++

    #return extractedWatermarksArray


def runAppropriateExtracMethod(pathToWatermarkedImageAfterAttack, method): #method = DWT.DWT_GRAY_LL
    #coverImagePath= "..."
    #watermarkImagePath = "..."
    #extractedWatermark
    # switch(Method)
        # case DWT.DWT_GRAY_LL
        #   extractedWatermark= DWT.DWT_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, pathToWatermarkedImageAfterAttack )
        #break
        # case DWT.DWT_SVD_GRAY_LL
        #   extractedWatermark = DWT.DWT_SVD_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, pathToWatermarkedImageAfterAttack ) #added _ExTRACT TO METHOD NAME
        #break
        #default ...

    #return extractedWatermark

def getExtractedWatermarkNormalWay(coverImagePath,watermarkImagePath, method):
    # extractedWatermark = runAppropriateImageWatermarkingMethod(coverImagePath,watermarkImagePath, method)
    # return extracted watermark

def runAppropriateImageWatermarkingMethod(method):  # method = DWT.DWT_GRAY_LL

    # extractedWatermark
    # switch(Method)
    # case DWT.DWT_GRAY_LL
    #   watermarkedImagePath= DWT.DWT_GRAY_LL(coverImagePath,watermarkImagePath) #no added _EXTRACT TO METHOD NAME
    #   extractedWatermark = DWT.DWT_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, watermarkedImagePath)
    # break
    # case DWT.DWT_SVD_GRAY_LL
    #   watermarkedImagePath= DWT_SVD.DWT_GRAY_LL(coverImagePath,watermarkImagePath) #no added _EXTRACT TO METHOD NAME
    #   extractedWatermark = DWT.DWT_SVD_GRAY_LL_EXTRACT(coverImagePath,watermarkImagePath, watermarkedImagePath)
    # break
    # default ...

    # return extractedWatermark

def compareWatermarks(watermarkAfterAttack, watermarkAfterNormalExtraction):

    #psnr(watermarkAfterAttack)
    #psnr(watermarkAfterNormalExtraction)
    #DO SMTH
