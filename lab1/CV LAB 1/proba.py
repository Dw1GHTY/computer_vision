from math import floor
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ol.jpg', 0)
plt.imshow(img, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.show()