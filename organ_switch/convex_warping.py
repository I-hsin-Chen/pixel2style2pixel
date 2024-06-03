import cv2
import numpy as np
from organ_switch.mask import get_organ_mask , get_landmark, left_eye, right_eye, align, get_organ_mask_mp, get_landmark_mp, align_mp, LeftEye, RightEye
import time

def floattoint(img):
    a = 255/(img.max() - img.min())
    b = 255 - a * img.max()
    new_img = (a * img + b).astype(np.uint8)
    return new_img

def affine_matrix(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])

def read_landmarks(img):
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    r, s = get_landmark_mp(img)
    return r, s

def warp_im(img, M, shape):
    output_img = np.zeros(shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (shape[1], shape[0]),
                   dst=output_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_img


def correct_colours(im1, im2, landmarks1):
    blur_amount = 0.6 * np.linalg.norm(
                              np.mean(landmarks1[LeftEye], axis=0) -
                              np.mean(landmarks1[RightEye], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)


    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    # im2_blur[im2_blur < 1] = 1
    correct_img = (im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64))
    correct_img = np.clip(correct_img, 0, 255)

    return correct_img


# for speeding up
def swap_organ_speed_up(source_path, target_path, tag, source_landmarks, target_landmarks, color_correct=True):

    source = cv2.resize(source_path, (source_path.shape[1], source_path.shape[0]))
    landmark1 = source_landmarks
    target = cv2.resize(target_path, (target_path.shape[1], target_path.shape[0]))
    landmark2 = target_landmarks

    M = affine_matrix(landmark1[align_mp], landmark2[align_mp])
    mask = get_organ_mask_mp(target, tag, landmark2)

    warp_mask = warp_im(mask, M, source.shape)
    combined_mask = np.max([get_organ_mask_mp(source, tag, landmark1), warp_mask],
                              axis=0)

    warp_target = warp_im(target, M, source.shape)
    if color_correct:
        correct_target = correct_colours(source, warp_target, landmark1)
    else:
        correct_target = warp_target
    output_img = source*(1.0-combined_mask) + correct_target * combined_mask
    
    import os
    # cv2.imwrite('mid_result/combined_mask.png', combined_mask * 255)
    # cv2.imwrite('mid_result/warp_target.png', warp_target.astype(np.uint8))
    # cv2.imwrite('mid_result/output_img.png', output_img.astype(np.uint8))
    # cv2.imwrite('mid_result/correct_target.png', correct_target.astype(np.uint8))

    return output_img.astype(np.uint8), correct_target.astype(np.uint8)


# the source get the target orgin
def swap_orgin(source_path, target_path, tag, source_is_read, source_landmarks):
    # speed up
    if source_is_read:
        source = cv2.resize(source_path, (source_path.shape[1], source_path.shape[0]))
        landmark1 = source_landmarks
    else:
        source, landmark1 = read_landmarks(source_path)

    target, landmark2 = read_landmarks(target_path)

    M = affine_matrix(landmark1[align], landmark2[align])

    mask = get_organ_mask_mp(target, tag, landmark2)
    warp_mask = warp_im(mask, M, source.shape)
    
    combined_mask = np.max([get_organ_mask_mp(source, tag, landmark1), warp_mask],
                              axis=0)

    warp_target = warp_im(target, M, source.shape)
    
    correct_target = correct_colours(source, warp_target, landmark1)
    output_img = source*(1.0-combined_mask) + correct_target*combined_mask

    return output_img, landmark1


if __name__ == '__main__':
    path1 = 'warping_sample/sample1/face.png'
    path2 = 'warping_sample/sample1/eyes.png'
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    out_img, s = swap_orgin(img1, img2, 'nose', False, [])
    cv2.imwrite('mouth.jpg', out_img)