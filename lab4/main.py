# python main.py -p "bvlc_googlenet.prototxt" -m "bvlc_googlenet.caffemodel" -l "synset_words.txt"

import numpy as np
import cv2
import imutils


def pyramid(img, scale, min_size):
    yield img
    while True:
        w = int(img.shape[1] / scale)
        img = imutils.resize(img, width=w)

        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break
        yield img


def crop_image(img, width, height):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    coords = [0, 0]
    dif_x = height
    dif_y = width

    for cont in contours:
        if cv2.contourArea(cont) >= width * height:
            arc_len = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)
            if len(approx) == 4:
                l_x, l_y = [], []

                for this_approx in approx:
                    cord_y = this_approx[0][0]
                    cord_x = this_approx[0][1]

                    l_y.append(cord_y)
                    l_x.append(cord_x)

                start_x = np.min(l_x)
                end_x = np.max(l_x)
                start_y = np.min(l_y)
                end_y = np.max(l_y)

                pom_dif_x = abs(height - (end_x - start_x))
                pom_dif_y = abs(width - (end_y - start_y))

                if pom_dif_x < dif_x and pom_dif_y < dif_y:
                    dif_x = pom_dif_x
                    dif_y = pom_dif_y
                    coords[0] = start_x
                    coords[1] = start_y
                    break

    return img[coords[0]:coords[0] + height, coords[1]:coords[1] + width]


def sliding_window(image, classes, net):
    for resized in pyramid(image, 2.0, (350, 100)):
        for y in range(0, resized.shape[0], step_size):
            for x in range(0, resized.shape[1], step_size):

                cropped_image = resized[y:y + window_size, x:x + window_size]
                blob = cv2.dnn.blobFromImage(cropped_image, 1, (224, 224), (104, 117, 123))
                net.setInput(blob)
                preds = net.forward()

                idxs = np.argsort(preds[0])[::-1][:1]
                idx = idxs[0]

                if preds[0][idx] > 0.5:
                    ratio = int(image.shape[1] / resized.shape[1])
                    x1 = x * ratio
                    y1 = y * ratio
                    region_width = window_size * ratio

                    if "dog" in classes[idx]:
                        color = (0, 255, 255)
                        text = "DOG"
                    elif "cat" in classes[idx]:
                        color = (0, 0, 255)
                        text = "CAT"
                    else:
                        continue
                    cv2.putText(image, text, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(image, (x1 + 2, y1 + 2), (x1 + region_width - 2, y1 + region_width - 2), color, 2)


window_size = 180
step_size = 180
width = 1440
height = 720

image = cv2.imread("input.png")
rows = open("synset_words.txt").read().strip().split("\n")
classes = [row[row.find(" ") + 1:].split(",")[0] for row in rows]
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

image = crop_image(image, width, height)
sliding_window(image, classes, net)
cv2.imwrite("output.jpg", image)
print("Fajl 'output.jpg' je sacuvan!")