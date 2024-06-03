import cv2
import os
import argparse
import time
import organ_switch.convex_warping as convex_warping
import organ_switch.triangles_warping as triangles_warping
from organ_switch.ffhq_encoder import face_reconstruction
from organ_switch.super_resolution import super_resolution
from organ_switch.threedface import get_3dresults
from organ_switch.convex_warping import read_landmarks


# ================== Load necessary files ==================
def load_files(path):
    with open('organ_landmarks.json', 'r') as f:
        import json
        tri_list = json.load(f)
    files = [os.path.join(path, f) for f in os.listdir(path)]

    images = {"face": None, "eyes": None, "mouth": None, "nose": None}
    flags = {"eyes": False, "mouth": False, "nose": False}
    landmarks = {}

    for file in files:
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        base_name = os.path.basename(file).split('.')[0]
        
        if base_name in images:
            images[base_name] = image
            if base_name in flags:
                flags[base_name] = True

    if images["face"] is None:
        raise AssertionError("Fail to load the face as basement!")
    _, landmarks["face"] = read_landmarks(images["face"])
    return images, flags, landmarks, tri_list

# ================== Convex warping ==================
def convex_all_organs(images, flags, landmarks):
    corrected_image = {}
    for organ in flags.keys():
        if flags[organ]:
            _, landmarks[organ] = read_landmarks(images[organ])
            images["face"], corrected_image[organ] = convex_warping.swap_organ_speed_up(\
                images["face"], images[organ], organ, landmarks["face"], landmarks[organ])
    return images, corrected_image


# ================== Traingles warping ==================
def triangle_all_organs(images, flags, tri_list, path):
    rough = {}
    rough["face"], _ = read_landmarks(images["face"])
    for organ in flags.keys():
        if flags[organ]:
            rough[organ], _ = read_landmarks(corrected_image[organ])
            images["face"] = triangles_warping.switch(images["face"], corrected_image[organ], tri_list[organ], rough["face"], rough[organ])
    cv2.imwrite(path + "/result.png", images["face"])
    return images["face"]


# ================== pixel2style2pixel + Super resolution ==================
def reconstruction2D(base_image, path):
    path = path + "/result.png"
    reconstructed_image = face_reconstruction(path)
    reconstructed_image.save(path)
    super_resolution(path)

    reconstructed_image = cv2.imread(path)
    result_image = base_image.copy()
    _, target_landmarks = read_landmarks(reconstructed_image)

    # Convex warping again
    for organ in ["eyes", "nose", "mouth"]:
        if flags[organ]:
            result_image, _ = convex_warping.swap_organ_speed_up(result_image, reconstructed_image, organ, landmarks["face"], target_landmarks, False)
    return result_image


if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to a directory containing images named face, eyes, mouth, nose', required=True)
    args = parser.parse_args()
    
    images, flags, landmarks, tri_list = load_files(args.path)
    base_image = images["face"]
    
    images, corrected_image = convex_all_organs(images, flags, landmarks)
    result_image = triangle_all_organs(images, flags, tri_list, args.path)
    result_image = reconstruction2D(base_image, args.path)
    get_3dresults(result_image, args.path)
    
    end = time.time()
    print('The whole process completed in : ' + format(end - start) + " seconds.")