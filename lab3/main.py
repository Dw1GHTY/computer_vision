import numpy as np
import cv2
from collections import deque


def merge(img1, img2):
    MIN_MATCH_COUNT = 10

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create()

    keypoint1, descriptor1 = detector.detectAndCompute(gray1, None)
    keypoint2, descriptor2 = detector.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        pts1 = np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        result_height = img1.shape[0] + img2.shape[0]
        result_width = img1.shape[1] + img2.shape[1]

        result = cv2.warpPerspective(img2, matrix, (result_width, result_height))
        res_cpy = result.copy()

        img1_height = img1.shape[0]
        img1_width = img1.shape[1]

        result[0:img1_height, 0:img1_width] = img1
        result = np.where(res_cpy == [0, 0, 0], result, res_cpy)
        non_black_pixels = np.where(result != [0, 0, 0])
        result = result[0:np.max(non_black_pixels[0]), 0:np.max(non_black_pixels[1])]
        return result
    else:
        raise Exception("Not enough matches are found!")


image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')
image3 = cv2.imread('img3.jpg')

result_img = image1
queue = deque()
queue.append(image2)
queue.append(image3)

try:
    while len(queue) > 0:
        next_img = queue.popleft()
        result_img = merge(result_img, next_img)

    cv2.imshow('result', result_img)
    cv2.imwrite('result.jpg', result_img)
    cv2.waitKey(0)
except Exception as ex:
    print(ex)
