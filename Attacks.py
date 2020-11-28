import calendar
import time

from PIL import Image


def compression(filename, quality):
    # https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/
    im = Image.open(filename)
    ts = calendar.timegm(time.gmtime())
    im.save("compressed/Compressed_" + quality + "_" + str(ts) + "_" + filename, optimize=True, quality=quality)
    return ''


def distorition(filename):
    # https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
    # https://www.geeksforgeeks.org/python-distort-method-in-wand/
    return ''


def transformation(filename):
    # https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/
    im = Image.open(filename)
    return ''
