import cv2
import numpy as np
import dlib
import mediapipe as mp

dlib_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

jaw_point = list(range(0, 17)) + list(range(68,81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))

# 裡面那一圈
LeftEye = [22, 23, 24, 25, 26, 27, 28, 29, 30, 56, 110, 112, 130, 190, 243, 247]
RightEye = [252, 253, 254, 255, 256, 257, 258, 259, 260, 286, 339, 359, 414, 463, 467]

# 外面那一圈
# LeftEye = [31, 113, 189, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 244]
# RightEye = [261, 342, 414, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453, 463]
Mouth = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405, 409]
Nose = [2, 8, 129, 188, 174, 193, 198, 209, 326, 460, 358, 399, 412, 417, 420, 429, 240, 102, 64]

align = (left_brow + right_eye + left_eye +
                               right_brow + nose + mouth)


def get_landmark(img):
    faces = detector(img, 1)
    shape = predictor(img, faces[0]).parts()
    return np.matrix([[p.x, p.y] for p in shape])

def get_landmark_mp(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
            landmarks = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).multi_face_landmarks[0]
    return landmarks


def draw_convex_hull(img, points, color):
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(img, hull, color=color)


def get_organ_mask(img, tag):
    landmarks = get_landmark(img)
    mask = np.zeros(img.shape[:2])
    if tag == 'eye':
        white = [right_eye, left_eye]
    if tag == 'nose':
        white = [nose]
    if tag == 'mouth':
        white = [mouth]
    if tag == 'eyebrow':
        white = [left_brow, right_brow]

    # print("white : " + format(white))
    for group in white:
        points = landmarks[group]
        # print(tag + " : " + format(points))
        draw_convex_hull(mask, points, 1)
    
    mask = np.array([mask]*3).transpose(1, 2, 0)
    mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask


def get_organ_mask_mp(img, tag):
    mask = np.zeros(img.shape[:2])
    if tag == 'eye':
        white = LeftEye
    if tag == 'nose':
        white = Nose
    if tag == 'mouth':
        white = Mouth

    landmarks = get_landmark_mp(img)
    points = []

    for num in white:
        point = [int(landmarks.landmark[num].x * img.shape[1]), int(landmarks.landmark[num].y * img.shape[0])]
        points.append(point)
    
    points = np.matrix([[p[0], p[1]] for p in points])
    draw_convex_hull(mask, points, 1)

    if (tag == 'eye'):
        white = RightEye
        points = []
        for num in white:
            point = [int(landmarks.landmark[num].x * img.shape[1]), int(landmarks.landmark[num].y * img.shape[0])]
            points.append(point)
    
        points = np.matrix([[p[0], p[1]] for p in points])
        draw_convex_hull(mask, points, 1)

    mask = np.array([mask]*3).transpose(1, 2, 0)
    mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask


def get_skin_mask(img):
    landmarks = get_landmark(img)
    mask = np.zeros(img.shape[:2])
    draw_convex_hull(mask, landmarks[jaw_point], color=1)
    for index in [mouth, left_eye, right_eye, left_brow, right_brow, nose]:
        draw_convex_hull(mask, landmarks[index], color=0)
    mask = np.array([mask] * 3).transpose(1, 2, 0)
    return mask


if __name__ == '__main__':
    path = '/home/pc/face_study/exp/timg.jpg'
    img = cv2.imread(path)
    mask = get_organ_mask(img, 'eye')
    cv2.imshow('a', mask)
    cv2.waitKey(0)