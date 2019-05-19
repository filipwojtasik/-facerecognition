import os
from scipy import ndimage, misc
import re
import matplotlib.pyplot as plt
import cv2
import glob

from PIL import Image
'''
image = ndimage.imread('face8.png', mode="RGB")
image_resized = misc.imresize(image, (200, 200))
cv2.imwrite('fff0.jpg' , cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
'''


def resizee():

    for j in range(75):
        images = []
        for root, dirnames, filenames in os.walk("zdjecia"):

            for filename in filenames:

                if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                    filepath = os.path.join(root, filename)
                    image = ndimage.imread(filepath, mode="RGB")
                    image_resized = misc.imresize(image, (200, 200))
                    images.append(image_resized)





    for i in range (len(images)):


        cv2.imwrite('zdj_resized/%(0)d.jpg' % {'0': i }, cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))







