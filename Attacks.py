
from PIL import Image
import PIL
class Attacks:

    def compression(self, filename, quality):
        # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
        im = Image.open(filename)
        im.save("Compressed" + filename, optimize=True, quality=quality)
        return ''

    def distorition(self, filename):
        # https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
        # https://www.geeksforgeeks.org/python-distort-method-in-wand/
        return ''

    def transformation(self, filename):
        # https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
        im = Image.open(filename)
        return ''
