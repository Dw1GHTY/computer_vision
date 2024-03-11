import numpy as np
import cv2

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel2 = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel2)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


imgIn = cv2.imread("coins.png")

imgHSV = cv2.cvtColor(imgIn, cv2.COLOR_BGR2HSV)

imgSat = imgHSV[:, :, 1]
imgSat[242, 331] = 30

imgGray = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)
imgGray2 = 255 - imgHSV[:, :, 1]

# def refresh(x):
#    _, imgOut = cv2.threshold(imgGray, x, 255, cv2.THRESH_BINARY_INV)
#    cv2.imshow("Grayscale thresh", imgOut)


# cv2.namedWindow("Saturation thresh")
# cv2.createTrackbar("tbTh", "Saturation thresh", 127, 255, refresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(11, 11))

_, imgOut = cv2.threshold(imgGray, 163, 255, cv2.THRESH_BINARY_INV)
imgClose = cv2.morphologyEx(imgOut, op=cv2.MORPH_CLOSE, kernel=kernel)

_, imgOutSat = cv2.threshold(imgGray2, 111, 255, cv2.THRESH_BINARY_INV)
imgSatClose = cv2.morphologyEx(imgOutSat, op=cv2.MORPH_CLOSE, kernel=kernel)

reconstructed = morphological_reconstruction(imgSatClose, imgClose)

cv2.imshow("Input", imgIn)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Close nad grayscale", imgClose)
# cv2.imshow("Saturation thresh", imgOutSat)
# cv2.imshow("Close nad saturation", imgSatClose)
cv2.imshow("Izdvojeni bakarni novcic", reconstructed)
cv2.waitKey(0)
