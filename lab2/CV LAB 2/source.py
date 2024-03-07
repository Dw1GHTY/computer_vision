import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

# Load the image
img = cv2.imread('coins.png')
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image
thresh_val = 133
ret, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

# Perform morphological opening to fill in any holes
opened = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get the saturation channel of the HSV image
sat = hsv[:,:,1]

# Threshold the saturation channel to segment the marker
marker_thresh_val = 29
ret, marker_thresh = cv2.threshold(sat, marker_thresh_val, 255, cv2.THRESH_BINARY)

# Perform morphological opening to remove any small noise
marker_open = cv2.morphologyEx(marker_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

# Perform morphological reconstruction to get the marker within the coin
marker = morphological_reconstruction(opened, marker_open)

bronze_coin = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask=marker) 

# Display the results
# plt.imshow(gray, cmap='gray')
# plt.show()
# plt.imshow(thresh, cmap='gray')
# plt.show()
# plt.imshow(opened, cmap='gray')
# plt.show()
# plt.imshow(marker_open, cmap='gray')
# plt.show()
plt.imshow(marker, cmap='gray')
plt.show()
plt.savefig('output_marker.png')
plt.imshow(bronze_coin, cmap='gray')
plt.show()