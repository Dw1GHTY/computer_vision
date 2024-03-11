import cv2 as cv
from cv2 import aruco
import numpy as np
import glob
from PIL import Image

BOARD_SIZE = (5, 7)
MARKER_SIZE = 2
MARKER_SEPARATION = 0.4
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
arucoParams = aruco.DetectorParameters()
board = aruco.GridBoard(BOARD_SIZE, MARKER_SIZE, MARKER_SEPARATION, aruco_dict)


def calibrate(files: str):
    counter, corners_list, id_list = [], [], []
    images = glob.glob(files)

    height, width = 0, 0

    for (iter, img) in enumerate(images):
        image = cv.imread(str(img))
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if iter == 0:
            height, width = image.shape[:2]
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))

    counter = np.array(counter)
    _, mtx, dist, _, _ = aruco.calibrateCameraAruco(corners_list,  id_list, counter, board, img_gray.shape, None, None)

    ncm, _ = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    return mtx, dist, ncm


def video(cap, camera_matrix, dist_coeffs, new_camera_matrix):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.array(frame)
        frame_remapped_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        corners, ids, rejected_img_points = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)
        aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejected_img_points)

        im_with_aruco_board = frame

        if type(ids) is np.ndarray and ids.any() is not None:
            im_with_aruco_board = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

            rvec = ()
            tvec = ()
            retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, rvec, tvec)
            if retval != 0:
                im_with_aruco_board = cv.drawFrameAxes(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, tvec, 5)

        undst = cv.undistort(im_with_aruco_board, camera_matrix, dist, None, new_camera_matrix)

        width = int(undst.shape[1] * 0.6)
        height = int(undst.shape[0] * 0.6)
        dim = (width, height)
        undst = cv.resize(undst, dim, interpolation=cv.INTER_AREA)

        cv.imshow("Output", undst)
        cv.imwrite('file.jpg', undst)
        if cv.waitKey(2) & 0xFF == 27:  # 27 - ESC character
            break

    return


mtx, dist, nmc = calibrate('Aruco/*.jpg')
cap = cv.VideoCapture("Aruco/Aruco_board.mp4")
video(cap, mtx, dist, nmc)
cap.release()
cv.destroyAllWindows()
