from concurrent.futures.process import _system_limited
# from msilib.schema import Error
from re import X
import string
import cv2
import mediapipe as mp
import numpy as np
import os
from organ_switch.convex_warping import correct_colours

def switch(base_image, target_image, Trilist, base_landmarks, target_landmarks):

    target_height,target_width,channels = base_image.shape
    warpped_destination = np.zeros((target_height, target_width, channels), np.uint8)
    lines_space_mask = np.zeros_like(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY))

    for tri in Trilist:

        (v1_x, v1_y) = int(base_landmarks.landmark[tri[0]].x * base_image.shape[1]), int(base_landmarks.landmark[tri[0]].y * base_image.shape[0])
        (v2_x, v2_y) = int(base_landmarks.landmark[tri[1]].x * base_image.shape[1]), int(base_landmarks.landmark[tri[1]].y * base_image.shape[0])
        (v3_x, v3_y) = int(base_landmarks.landmark[tri[2]].x * base_image.shape[1]), int(base_landmarks.landmark[tri[2]].y * base_image.shape[0])

        (target_v1_x, target_v1_y) = int(target_landmarks.landmark[tri[0]].x * target_image.shape[1]), int(target_landmarks.landmark[tri[0]].y * target_image.shape[0])
        (target_v2_x, target_v2_y) = int(target_landmarks.landmark[tri[1]].x * target_image.shape[1]), int(target_landmarks.landmark[tri[1]].y * target_image.shape[0])
        (target_v3_x, target_v3_y) = int(target_landmarks.landmark[tri[2]].x * target_image.shape[1]), int(target_landmarks.landmark[tri[2]].y * target_image.shape[0])

        base_tri = np.array([[v1_x, v1_y], [v2_x, v2_y], [v3_x, v3_y]], np.int32)
        target_tri = np.array([[target_v1_x, target_v1_y], [target_v2_x, target_v2_y], [target_v3_x, target_v3_y]], np.int32)
        cv2.fillConvexPoly(base_image, base_tri, 255)

        # target image triangle
        rect1 = cv2.boundingRect(target_tri)
        (x, y, w, h) = rect1
        cropped_triangle = target_image[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        
        points = np.array([[target_v1_x - x, target_v1_y - y],
                        [target_v2_x - x, target_v2_y - y],
                        [target_v3_x - x, target_v3_y - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
        lines_space = cv2.bitwise_and(target_image, target_image, mask=lines_space_mask)

        # base image triangle
        rect2 = cv2.boundingRect(base_tri)
        (x, y, w, h) = rect2
        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[v1_x - x, v1_y - y],
                            [v2_x - x, v2_y - y],
                            [v3_x - x, v3_y - y]], np.int32)
        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        img2_new_face_rect_area = warpped_destination[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        warpped_destination[y: y + h, x: x + w] = img2_new_face_rect_area

    img = cv2.cvtColor(warpped_destination, cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV)
    paste_area = cv2.bitwise_and(base_image, base_image, mask=mask)
    cv2.imwrite('mid_result/paste_area.png', paste_area)
    cv2.imwrite('mid_result/warpped_destination.png', warpped_destination)
    result = cv2.add(paste_area, warpped_destination)
    return result