import numpy as np
import cv2
import pywt

def SVD_GRAY_LL():
    coverImage = cv2.imread('mandrill.jpg', 0)
    watermarkImage = cv2.imread('lenna_256.jpg', 0)

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
    S = np.zeros((m, n), np.uint8)  # change for 512 from 225
    S[:m, :n] = np.diag(w)
    S = np.double(S)
    wimg = np.matmul(ucvr, np.matmul(S, vtcvr))  # np.matmul- function returns the matrix product of two arrays
    wimg = np.double(wimg)
    wimg *= 255
    watermarkedImage = np.zeros(wimg.shape, np.double)
    normalized = cv2.normalize(wimg, watermarkedImage, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow('Watermarked Image', watermarkedImage)

    # cv2.imshow('Watermarked Image', wimgr)



if __name__ == "__main__":

    options = {1: SVD_GRAY_LL
               }
    val = int(input('What type of embedding you want to perform?\n1.SVD_GRAY_EMBED '))
    options[val]()

    cv2.waitKey(0)
    cv2.destroyAllWindows()