import numpy as np
import cv2
import math

def DCT2(coverImage, watermarkImage):
    coverImage = cv2.imread(coverImage, 0)
    watermarkImage = cv2.imread(watermarkImage, 0)
    coverImage = cv2.resize(coverImage,(512,512))
    cv2.imshow('Cover Image',coverImage)
    watermarkImage = cv2.resize(watermarkImage,(64,64))
    cv2.imshow('Watermark Image',watermarkImage)
    
    coverImage =  np.float32(coverImage)   
    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255

    blockSize = 8
    c1 = np.size(coverImage, 0)
    c2 = np.size(coverImage, 1)
    max_message = int((c1*c2)/(blockSize*blockSize))

    w1 = int(np.size(watermarkImage, 0))
    w2 = int(np.size(watermarkImage, 1))

    watermarkImage = np.round(np.reshape(watermarkImage,(w1*w2, 1)),0)

    if w1*w2 > max_message:
        print ('Message too large to fit')

    message_pad = np.ones((max_message,1), np.float32)
    message_pad[0:w1*w2] = watermarkImage

    watermarkedImage = np.ones((c1,c2), np.float32)

    k=50
    a=0
    b=0

    for kk in range(max_message):
        dct_block = cv2.dct(coverImage[b:b+blockSize, a:a+blockSize])
        if message_pad[kk] == 0:
            if dct_block[4,1]<dct_block[3,2]:
                temp=dct_block[3,2]
                dct_block[3,2]=dct_block[4,1]
                dct_block[4,1]=temp
        else:
            if dct_block[4,1]>=dct_block[3,2]:
                temp=dct_block[3,2]
                dct_block[3,2]=dct_block[4,1]
                dct_block[4,1]=temp

        if dct_block[4,1]>dct_block[3,2]:
            if dct_block[4,1] - dct_block[3,2] <k:
                dct_block[4,1] = dct_block[4,1]+k/2
                dct_block[3,2] = dct_block[3,2]-k/2
        else:
            if dct_block[3,2] - dct_block[4,1]<k:
                dct_block[3,2] = dct_block[3,2]+k/2
                dct_block[4,1] = dct_block[4,1]-k/2
            
        watermarkedImage[b:b+blockSize, a:a+blockSize]=cv2.idct(dct_block)
        if a+blockSize>=c1-1:
            a=0
            b=b+blockSize
        else:
            a=a+blockSize

    watermarkedImage_8 = np.uint8(watermarkedImage)
    cv2.imshow('watermarked',watermarkedImage_8)
    cv2.waitKey(0)


def DCT (coverImage, watermarkImage):
    cImage = cv2.imread(coverImage)
    cImage = np.float32(cImage) / 255
    wImage = cv2.imread(watermarkImage)
    wImage = np.float32(wImage) / 255

    try:
        watermarkHeight, watermarkWidth = wImage.shape[:2]
        coverImageHeight, coverImageWidth = cImage.shape[:2]
        watermarkStrength = 1

        if watermarkHeight > coverImageHeight or watermarkWidth > coverImageWidth:
            raise Exception()

        print('Watermark size: ', watermarkHeight, ' x ', watermarkWidth)
        print('Image size: ', coverImageHeight, ' x ', coverImageWidth)

    except:
        print('Watermark image must be smaller than cover image!')

    heightDiff = coverImageHeight - watermarkHeight
    widthDiff = coverImageWidth - watermarkWidth

    wImage = cv2.copyMakeBorder(
        wImage, 
        top = math.floor(heightDiff / 2),
        bottom = math.ceil(heightDiff / 2),
        left = math.floor(widthDiff / 2),
        right = math.ceil(widthDiff / 2),
        borderType = 0
    )

    (cb, cg, cr) = cv2.split(cImage)

    CdctB = cv2.dct(cb)
    CdctG = cv2.dct(cg)
    CdctR = cv2.dct(cr)

    (wb, wg, wr) = cv2.split(wImage)

    CdctB += wb * watermarkStrength
    CdctG += wg * watermarkStrength
    CdctR += wr * watermarkStrength

    finalB = cv2.idct(CdctB)
    finalG = cv2.idct(CdctG)
    finalR = cv2.idct(CdctR)

    finalImage = cv2.merge((finalB, finalG, finalR))
    diffImage = cImage - finalImage

    cv2.imshow('original image', cImage)
    cv2.imshow('watermarked image', finalImage)
    cv2.imshow('difference', diffImage)
    cv2.waitKey(0)


DCT2(coverImage='images/cat.jpg', watermarkImage='images/lena.jpg')
DCT(coverImage='images/cat.jpg', watermarkImage='images/lena.jpg')
