import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

newFolder = "Output files"

if not os.path.exists(newFolder):
    os.makedirs(newFolder)

loadedImage = cv.imread("slika_0.png")
inputImage = cv.cvtColor(loadedImage, cv.COLOR_BGR2GRAY)

cv.imwrite(os.path.join(newFolder, '1 Ulazna slika.png'), inputImage)

plt.subplot(221)
plt.imshow(inputImage, cmap='gray')
plt.title("Ulazna slika")

inputImage_fft = np.fft.fft2(inputImage)
inputImage_fft_shifted = np.fft.fftshift(inputImage_fft)
inputImage_magnitude = np.abs(inputImage_fft_shifted)

inputImage_magnitude_1 = inputImage_fft_shifted / inputImage_magnitude

inputImage_magnitude = np.log(inputImage_magnitude)

cv.imwrite(os.path.join(newFolder, '2 Magnituda spektra pre uklanjanja suma.png'), inputImage_magnitude)

plt.subplot(222)
plt.imshow(inputImage_magnitude, cmap='gray')
plt.title("Magnituda spektra pre uklanjanja šuma")

inputImage_magnitude[156, 236] = 0
inputImage_magnitude[156, 276] = 0
inputImage_magnitude[356, 276] = 0
inputImage_magnitude[356, 236] = 0

cv.imwrite(os.path.join(newFolder, '3 Magnituda spektra nakon uklanjanja suma.png'), inputImage_magnitude)

plt.subplot(223)
plt.imshow(inputImage_magnitude, cmap='gray')
plt.title("Magnituda spektra nakon uklanjanja šuma")

inputImage_unshifted = inputImage_magnitude_1 * np.exp(inputImage_magnitude)
inputImage_ifft = np.fft.ifft2(inputImage_unshifted)
inputImage_restored = np.abs(inputImage_ifft)

cv.imwrite(os.path.join(newFolder, '4 Ulazna slika nakon uklanjanja suma.png'), inputImage_restored)

plt.subplot(224)
plt.imshow(inputImage_restored, cmap='gray')
plt.title("Ulazna slika nakon uklanjanja šuma")

plt.show()

