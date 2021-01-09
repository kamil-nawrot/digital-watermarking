import cv2
import numpy as np
import pywt


class DWTDCT:
    def __init__(self, name, path, dim=None, mode=8):
        self.name = name
        img = cv2.imread(path, mode)
        self.img = img
        if (dim == None):
            self.dim = np.shape(self.img)[:2]
            self.dim = self.dim[::-1]
        else:
            self.dim = dim
        if self.dim[0] % 8 != 0:
            self.dim = (self.dim[0] + (self.dim[0] % 8), self.dim[1])
        if self.dim[1] % 8 != 0:
            self.dim = (self.dim[0], self.dim[1] + (self.dim[1] % 8))
        self.img = cv2.resize(self.img, self.dim)
        self.img = np.float32(self.img) / 255
        self.channels = cv2.split(self.img)
        self.coeffs = None

    
    def display(self):
        cv2.imshow(self.name, self.img)
        cv2.waitKey(0)


    def display_difference(self, referenceImage):
        if isinstance(referenceImage, DWTDCT):
            cv2.imshow('Difference', self.img - referenceImage.img)
            cv2.waitKey(0)
            return


    def save(self, path):
        img = np.clip(self.img * 255, 0, 255)
        img = np.uint8(img)
        cv2.imwrite(path,  img)
        return

    def calculate_coefficients(self):
        coeffs = []
        for c in self.channels:
            coeffs.append(pywt.dwtn(c, wavelet="haar"))
        self.coeffs = coeffs
        return coeffs


    def apply_dct(self, dwtSet):
        setDict = {'LL': 'aa', 'LH': 'ad', 'HL': 'da', 'HH': 'dd'}
        self.subset = setDict[dwtSet]
        i = 0
        for c in self.coeffs:
            coeffSubset = c[self.subset]
            dctChannel = np.empty(np.shape(coeffSubset))
            for r in range(0, len(dctChannel), 4):
                for p in range(0, len(dctChannel[0]), 4):
                    block = coeffSubset[r:r+4, p:p+4]
                    dctBlock = cv2.dct(block)
                    dctChannel[r:r+4, p:p+4] = dctBlock
            self.coeffs[i][self.subset] = dctChannel
            i += 1
        return self.coeffs


    def invert_dct(self):
        i = 0
        for c in self.coeffs:
            coeffSubset = c[self.subset]
            idctChannel = np.empty(np.shape(coeffSubset))
            for r in range(0, len(idctChannel), 4):
                for p in range(0, len(idctChannel[0]), 4):
                    block = coeffSubset[r:r+4, p:p+4]
                    idctBlock = cv2.idct(block)
                    idctChannel[r:r+4, p:p+4] = idctBlock
            self.coeffs[i][self.subset] = idctChannel
            i += 1

        return self.coeffs


    def embed_watermark(self, dwtSet, watermarkImage):
        self.calculate_coefficients()
        self.apply_dct(dwtSet)
        vectors = list(map(lambda c: np.ravel(c), watermarkImage.channels))
        for c in range(len(self.channels)):
            i = 0
            coeffSubset = self.coeffs[c][self.subset]
            for r in range(0, len(coeffSubset), 4):
                for p in range(0, len(coeffSubset[0]), 4):
                    if i < len(vectors[c]):
                        dctBlock = coeffSubset[r:r+4, p:p+4]
                        dctBlock[2][2] = vectors[c][i]
                        coeffSubset[r:r+4, p:p+4] = dctBlock
                        i += 1
            self.coeffs[c][self.subset] = coeffSubset
        self.invert_dct()
        reconstructed = self.reconstruct()
        return reconstructed
    

    def reconstruct(self):
        reconstructedImage = []
        for c in range(len(self.channels)):
            reconstructedChannel = pywt.idwtn(self.coeffs[c], wavelet="haar")
            reconstructedImage.append(reconstructedChannel)

        reconstructed = cv2.merge(reconstructedImage)
        self.img = reconstructed
        self.channels = cv2.split(reconstructed)
        cv2.imshow("Watermarked Image", reconstructed)
        cv2.waitKey(0)
        return reconstructed

    
    def extract_watermark(self, dwtSet, watermarkSize):
        self.calculate_coefficients()
        self.apply_dct(dwtSet)

        watermark = []

        for c in range(len(self.channels)):
            coeffSubset = self.coeffs[c][self.subset]
            watermarkChannel = []
            i = 0
            for r in range(0, len(coeffSubset), 4):
                for p in range(0, len(coeffSubset[0]), 4):
                    if (i < watermarkSize*watermarkSize):
                        block = coeffSubset[r:r+4, p:p+4]
                        watermarkChannel.append(block[2][2])
                        i += 1
            watermark.append(watermarkChannel)
        watermark = np.reshape(watermark, (len(self.channels), watermarkSize, watermarkSize,))
        watermark = cv2.merge(watermark)
        cv2.imshow("Extracted Watermark", watermark)
        cv2.waitKey(0)

        return


# baseImage = Image("base", "images/lenna_256.jpg", (1024, 1024))
# originalImage = Image("base", "images/lenna_256.jpg", (1024, 1024))
# watermarkImage = Image("watermark", "images/mandrill_256.jpg", (128, 128))
#
# baseImage.embed_watermark('HL', watermarkImage)
# baseImage.display()
# baseImage.display_difference(originalImage)
# baseImage.save('watermarked_image.jpg')
#
# reconstructedImage = Image("watermarked", "watermarked_image.jpg")
# reconstructedImage.extract_watermark('HL', 128)
