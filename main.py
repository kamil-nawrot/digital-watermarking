import cv2
import numpy as np

import DWT as DWT
from attacks import Attacks as Attacks


def SVD(coverImage, watermarkImage):
    cv2.imshow('Cover Image', coverImage)
    [m, n] = np.shape(coverImage)
    coverImage = np.double(coverImage)
    cv2.imshow('Watermark Image', watermarkImage)
    watermarkImage = np.double(watermarkImage)

    # SVD of cover image
    ucvr, wcvr, vtcvr = np.linalg.svd(coverImage, full_matrices=1, compute_uv=1)
    Wcvr = np.zeros((m, n), np.uint8)
    Wcvr[:m, :n] = np.diag(wcvr)
    Wcvr = np.double(Wcvr)
    [x, y] = np.shape(watermarkImage)

    # modifying diagonal component
    for i in range(0, x):
        for j in range(0, y):
            Wcvr[i, j] = (Wcvr[i, j] + 0.01 * watermarkImage[i, j]) / 255

    # SVD of wcvr
    u, w, v = np.linalg.svd(Wcvr, full_matrices=1, compute_uv=1)

    # Watermarked Image
    S = np.zeros((m, n), np.uint8) #change for 512 from 225
    S[:m, :n] = np.diag(w)
    S = np.double(S)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr)) #  np.matmul- function returns the matrix product of two arrays
    wimg = np.double(wimg)
    wimg *= 255
    watermarkedImage = np.zeros(wimg.shape, np.double)
    normalized = cv2.normalize(wimg, watermarkedImage, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow('Watermarked Image', watermarkedImage)


def SVD():
    return


def DWT_SVD():
    return


def DWT_DCT():
    return


def compression(quality):
    return Attacks.compression("lenna_256.jpg", quality)

if __name__ == "__main__":
    coverImage = cv2.imread('mandrill.jpg', 0)
    watermarkImage = cv2.imread('lenna_256.jpg', 0)

    options = {1: "DWT",
               2: SVD,
               3: DWT_SVD,
               4: DWT_DCT,
               5: compression
               }

    choice = int(input('What type of embedding you want to perform?\n1.DWT\n2.SVD\n3.SVD-DWT\n4.Compression\n'))
    if choice == 1:
        opt = int(input('What type of embedding you want to perform?\n1.DWT_GRAY_EMBED\n2.DWT_RGB_LL\n'))
        if opt == 1:
            DWT.DWT_RGB_LL("mandrill.jpg", "lenna.jpg")
        elif opt == 2:
            DWT.DWT_GRAY_LL("mandrill.jpg", "lenna.jpg")
    elif choice > 1 and choice < 4:
        options[choice](coverImage, watermarkImage)
    elif choice == 5:
        quality = int(input('Compression quality <0, 100>\n'))
        options[choice](quality)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
