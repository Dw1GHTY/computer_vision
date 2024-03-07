import cv2
import numpy as np
import matplotlib.pyplot as plt

#DRUGO RESENJE
img = cv2.imread('coins.png')

# Convert the input image to the HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Extract the S channel from the HSV color space
s_channel = hsv[:,:,1]

# Apply adaptive thresholding to the S channel
thresh = cv2.adaptiveThreshold(s_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
# Find the contours in the thresholded image
contours, _ = cv2.findContours(thresh_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_mean = 0
max_contour = None
for contour in contours:
    # Compute the mean intensity value of the S channel within the contour
    mask = np.zeros_like(s_channel)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    mean_val = cv2.mean(s_channel, mask=mask)[0]

    # Update the maximum mean intensity value and contour if necessary
    if mean_val > max_mean:
        max_mean = mean_val
        max_contour = contour


# Draw the brightest contour on the original image
((x, y), radius) = cv2.minEnclosingCircle(max_contour)

# Create a binary mask with ones inside the circle and zeros outside
scaling_factor = 0.8
mask = np.zeros_like(s_channel)
cv2.circle(mask, (int(x), int(y)),  int(scaling_factor * radius), (255, 255, 255), -1)

bronze_coin = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask=mask) 
# Display the output image
plt.imshow(s_channel)
plt.show()
plt.imshow(thresh)
plt.show()
plt.imshow(mask)
plt.show()
plt.imshow(bronze_coin)
plt.show()
