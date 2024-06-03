from concurrent.futures.process import _system_limited
from email.mime import base
# from msilib.schema import Error
from re import X
import string
from tkinter.tix import Tree
from unittest import result
import cv2
import mediapipe as mp
import numpy as np
import os
from organ_switch.mask import get_organ_mask , get_landmark, left_eye, right_eye, align
import organ_switch.convex_warping as convex_warping
import organ_switch.triangles_warping as triangles_warping
import organ_switch.ffhq_encoder as ffhq_encoder
import organ_switch.super_resolution as super_resolution
import organ_switch.threedface as threedface

import time

# change "path" to the image directory
path = "warping_sample/sample4"
eyes = False
mouth = False
nose = False


LeftEyeTriList = [(33,130,247),(25,33,130), (25,7,33), (7,33,246), (33,246,247), (161,246,247), (30,161,247), (7,25,110), (7,163,246), (7,110,163), (110,144,163), (24,110,144), (161,163,246), (144,161,163), (30,160,161), (29,30,160), (29,159,160), (27,29,159), (27,28,159), (28,158,159), (28,56,158), (56,157,158), (56,157,190), (157,173,190), (173,190,243), (133,173,243), (112,133,243), (112,133,155), (133,155,173), (26,112,155), (155,157,173), (26,154,155), (154,155,157), (154,157,158), (22,26,154), (22,153, 154), (153,154,158), (145,153,159), (22,23,153), (23,145,153), (144,145,160), (144,160,161), (23,144,145), (23,24,144), (145,159,160),(153,158,159)]
RightEyeTriList = [(398,414,463), (362,398,463), (341,362,463), (362,382,398), (341,362,382), (384,398,414), (286,384,414), (382,384,398), (256,341,382), (256,381,382), (381,382,384), (286,384,385), (258,286,385), (381,384,385), (380,381,385), (252,380,381), (252,256,381), (252,253,380), (253,374,380), (374,380,386), (380,385,386), (258,385,386), (257,258,386), (257,259,386), (259,386,387), (374,386,387), (373,374,387), (253,373,374), (253,254,373), (254,339,373), (339,373,390), (373,388,390), (373,387,388), (260,387,388), (259,260,387), (260,388,467), (388,466,467), (388,390,466), (249,390,466), (249,339,390), (249,255,339), (249,255,263), (249,263,466), (263,466,467), (263,359,467), (255,263,359)]
#NoseTriList = [(290,309,392),(8,55,193),(8,285,417),(8,168,193),(8,168,417),(122,168,193),(168,351,417),(6,122,168),(6,168,351),(351,465,417),(122,193,245),(122,188,245),(351,412,465),(122,188,196),(6,122,196),(6,196,197),(6,197,419),(6,351,419),(412,419,351),(399,412,419),(174,188,196),(174,198,236),(174,196,236),(3,196,236),(3,196,197),(3,195,197),(195,197,248),(197,248,419),(248,456,419),(399,419,456),(399,420,456),(279,358,429),(279,420,429),(360,279,420),(360,363,420),(363,420,456),(363,420,456),(281,363,456),(248,281,456),(195,248,281),(5,195,281),(5,51,195),(3,51,195),(3,51,236),(51,134,236),(134,198,236),(134,198,236),(131,134,198),(49,131,198),(49,198,209),(49,129,209),(49,102,129),(48,49,129),(48,49,131),(48,115,131),(115,131,220),(131,134,220),(45,134,220),(45,51,134),(5,45,51),(4,5,45),(4,5,275),(5,275,281),(275,281,363),(275,363,440),(360,363,440),(344,360,440),(278,344,360),(278,279,360),(278,279,331),(279,331,358),(294,331,358),(294,327,358),(278,294,331),(278,294,439),(278,344,439),(344,438,439),(344,438,440),(438,440,457),(274,457,440),(274,275,440),(1,274,275),(1,4,275),(1,4,45),(1,44,45),(44,45,220),(44,220,237),(218,220,237),(115,218,220),(115,218,219),(48,115,219),(48,64,219),(48,64,102),(64,102,129),(64,102,129),(64,98,129),(64,98,240),(64,235,240),(64,219,235),(75,235,240),(59,75,235),(59,219,235),(59,166,219),(166,218,219),(79,166,218),(79,218,239),(218,237,239),(237,239,241),(44,237,241),(44,125,241),(19,44,125),(1,19,44),(1,19,274),(19,274,354),(274,354,461),(274,457,461),(457,459,461),(458,459,461),(438,457,459),(309,438,459),(309,458,459),(309,392,438),(392,438,439),(289,392,439),(439,289,455),(294,439,455),(294,327,460),(294,455,460),(305,455,460),(305,289,455),(289,305,392),(290,305,392),(290,305,392),(250,290,309),(250,309,458),(250,458,462),(458,461,462),(370,461,462),(354,370,461),(94,354,370),(19,94,354),(19,94,125),(94,125,141),(125,141,241),(141,241,242),(238,241,242),(20,238,242),(20,79,238),(79,238,239),(20,60,79),(60,79,166),(60,75,166),(59,75,166),(60,75,99),(75,99,240),(97,99,240),(240,97,98),(20,60,99),(20,99,242),(97,99,242),(97,141,242),(2,97,141),(2,94,141),(2,94,370),(2,326,370),(326,370,462),(326,328,462),(250,328,462),(250,290,328),(290,305,328),(305,328,460),(326,328,460),(326,327,460),(114,188,245),(114,174,188),(114,174,217),(174,198,217)]
MouthTriList = [(61,76,185),(61,76,146),(62,76,183),(76,183,184),(76,184,185),(62,76,77),(76,77,146),(62,183,191),(62,95,96),(62,77,96),(77,91,146),(77,90,91),(77,90,96),(89,90,96),(88,89,96),(88,95,96),(80,183,191),(42,80,183),(42,183,184),(42,74,184),(40,74,184),(40,184,185),(40,73,74),(39,40,73),(41,73,74),(41,42,74),(41,42,81),(42,80,81),(88,178,179),(88,89,179),(89,90,179),(90,179,180),(90,91,180),(91,180,181),(84,85,181),(85,180,181),(85,86,180),(86,179,180),(86,87,179),(87,178,179),(41,81,82),(38,41,82),(38,41,73),(38,72,73),(39,72,73),(39,72,73),(37,39,72),(0,37,72),(0,11,72),(11,12,72),(12,38,72),(12,13,38),(13,38,82),(14,86,87),(14,15,86),(15,85,86),(15,16,85),(16,84,85),(16,17,84),(0,11,302),(11,12,302),(12,268,302),(0,267,302),(12,13,268),(13,268,312),(14,316,317),(14,15,316),(15,315,316),(15,16,315),(16,314,315),(16,17,314),(314,315,405),(315,404,405),(315,316,404),(316,403,404),(316,317,403),(317,402,403),(271,311,312),(268,271,312),(268,271,303),(268,302,303),(269,302,303),(267,269,302),(269,270,303),(270,303,304),(271,303,304),(271,272,304),(271,272,311),(272,310,311),(318,402,403),(318,319,403),(319,320,403),(320,403,404),(320,321,404),(321,404,405),(307,320,321),(307,321,375),(307,320,325),(319,320,325),(318,319,325),(318,324,325),(310,407,415),(272,310,407),(272,407,408),(272,304,408),(270,304,408),(270,408,409),(292,407,408),(408,409,306),(291,306,409),(292,306,407),(292,407,415),(292,324,325),(292,307,325),(292,306,307),(306,307,375),(291,306,375)]
NoseTriList = [(290,309,392),(122,188,196),(6,122,196),(6,196,197),(6,197,419),(6,351,419),(412,419,351),(399,412,419),(174,188,196),(174,198,236),(174,196,236),(3,196,236),(3,196,197),(3,195,197),(195,197,248),(197,248,419),(248,456,419),(399,419,456),(399,420,456),(279,358,429),(279,420,429),(360,279,420),(360,363,420),(363,420,456),(363,420,456),(281,363,456),(248,281,456),(195,248,281),(5,195,281),(5,51,195),(3,51,195),(3,51,236),(51,134,236),(134,198,236),(134,198,236),(131,134,198),(49,131,198),(49,198,209),(49,129,209),(49,102,129),(48,49,129),(48,49,131),(48,115,131),(115,131,220),(131,134,220),(45,134,220),(45,51,134),(5,45,51),(4,5,45),(4,5,275),(5,275,281),(275,281,363),(275,363,440),(360,363,440),(344,360,440),(278,344,360),(278,279,360),(278,279,331),(279,331,358),(294,331,358),(294,327,358),(278,294,331),(278,294,439),(278,344,439),(344,438,439),(344,438,440),(438,440,457),(274,457,440),(274,275,440),(1,274,275),(1,4,275),(1,4,45),(1,44,45),(44,45,220),(44,220,237),(218,220,237),(115,218,220),(115,218,219),(48,115,219),(48,64,219),(48,64,102),(64,102,129),(64,102,129),(64,98,129),(64,98,240),(64,235,240),(64,219,235),(75,235,240),(59,75,235),(59,219,235),(59,166,219),(166,218,219),(79,166,218),(79,218,239),(218,237,239),(237,239,241),(44,237,241),(44,125,241),(19,44,125),(1,19,44),(1,19,274),(19,274,354),(274,354,461),(274,457,461),(457,459,461),(458,459,461),(438,457,459),(309,438,459),(309,458,459),(309,392,438),(392,438,439),(289,392,439),(439,289,455),(294,439,455),(294,327,460),(294,455,460),(305,455,460),(305,289,455),(289,305,392),(290,305,392),(290,305,392),(250,290,309),(250,309,458),(250,458,462),(458,461,462),(370,461,462),(354,370,461),(94,354,370),(19,94,354),(19,94,125),(94,125,141),(125,141,241),(141,241,242),(238,241,242),(20,238,242),(20,79,238),(79,238,239),(20,60,79),(60,79,166),(60,75,166),(59,75,166),(60,75,99),(75,99,240),(97,99,240),(240,97,98),(20,60,99),(20,99,242),(97,99,242),(97,141,242),(2,97,141),(2,94,141),(2,94,370),(2,326,370),(326,370,462),(326,328,462),(250,328,462),(250,290,328),(290,305,328),(305,328,460),(326,328,460),(326,327,460),(174,198,217)]

# debug arguments
#BUGSFIXED_glitch = True
time_cnt = True

files = [os.path.join(path, f) for f in os.listdir(path)]
start = time.time()

for file in files:
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    str = os.path.basename(file).split('.')[0]
    if str == "face": 
        tmp_image = image
        base_image = image
    if str == "eyes": 
        eyes_image = image
        eyes = True
    if str == "mouth":
        mouth_image = image
        mouth = True
    if str == "nose":
        nose_image = image
        nose = True

if 'base_image' not in globals():
    raise AssertionError("Fail to load the face as basement !")

file_time = time.time()

# For speed-up sake, get all the landmarks first
_, base_landmarks = convex_warping.read_landmarks(base_image)

# Do convex_warping
if eyes != False:
    rough_eye, eyes_landmarks = convex_warping.read_landmarks(eyes_image)
    base_image, correct_eye_image = convex_warping.swap_organ_speed_up(base_image, eyes_image, "eye", base_landmarks, eyes_landmarks)

if nose != False:
    rough_nose, nose_landmarks = convex_warping.read_landmarks(nose_image)
    base_image, correct_nose_image = convex_warping.swap_organ_speed_up(base_image, nose_image, "nose", base_landmarks, nose_landmarks)

if mouth != False:
    rough_mouth, mouth_landmarks = convex_warping.read_landmarks(mouth_image)
    base_image, correct_mouth_image = convex_warping.swap_organ_speed_up(base_image, mouth_image, "mouth", base_landmarks, mouth_landmarks)

after_cvx = base_image.copy()
cvx1_time = time.time()
cv2.imwrite(path + "/after_cvx.png", after_cvx)

# and then do triangles warping
rough_base, landmarks_after_cvx = convex_warping.read_landmarks(base_image)
if eyes != False:
    EyeTriList = LeftEyeTriList + RightEyeTriList
    rough_eye, eyes_landmarks = convex_warping.read_landmarks(correct_eye_image)
    base_image = triangles_warping.switch(base_image, correct_eye_image, EyeTriList, rough_base, rough_eye)
if nose != False:
    rough_nose, nose_landmarks = convex_warping.read_landmarks(correct_nose_image)
    base_image = triangles_warping.switch(base_image, correct_nose_image, NoseTriList, rough_base, rough_nose)
if mouth != False:
    rough_mouth, mouth_landmarks = convex_warping.read_landmarks(correct_mouth_image)
    base_image = triangles_warping.switch(base_image, correct_mouth_image, MouthTriList, rough_base, rough_mouth)

cv2.imwrite(path + "/result.png", base_image)
cv2.imwrite(path + "/after_tri.png", base_image)

tri_time = time.time()

# Input pixel2style2pixel ffhq_encoder
reconstructed_image = ffhq_encoder.face_reconstruction(path + "/result.png", False)
reconstructed_image.save(path + "/result.png")
reconstructed_image.save("mid_result/after_psp.png")

psp_time = time.time()

# Do super resolution
super_resolution.super_resolution(path + "/result.png")
sr_time = time.time()

# Do convex warping again
reconstructed_image = cv2.imread(path + "/result.png")
result_image = tmp_image
_, target_landmarks = convex_warping.read_landmarks(reconstructed_image)

if mouth != False:
    result_image, _ = convex_warping.swap_organ_speed_up(result_image, reconstructed_image, "mouth", base_landmarks, target_landmarks, False)

if eyes != False:
    result_image, _ = convex_warping.swap_organ_speed_up(result_image, reconstructed_image, "eye", base_landmarks, target_landmarks, False)

if nose != False:
    result_image, _ = convex_warping.swap_organ_speed_up(result_image, reconstructed_image, "nose", base_landmarks, target_landmarks, False)

# cv2.imwrite(path + "/tmp.png", result_image)

td_time = time.time()
threedface.get_3dresults(result_image, path)

end = time.time()

if time_cnt:
    print('The whole process completed in : ' + format(end - start) + " seconds.")
    # print(' file_time : ' + format(file_time - start) + ' s.')
    # print(' cvx1_time : ' + format(cvx1_time - file_time) + ' s.')
    # print(' tri_time : ' + format(tri_time - cvx1_time) + ' s.')
    # print(' psp_time : ' + format(psp_time - tri_time) + ' s.')
    # print(' sr_time : ' + format(sr_time - psp_time) + ' s.')
    # print(' cvx2_time : ' + format(td_time - sr_time) + ' s.')
    # print(' 3d_time : ' + format(end - td_time) + ' s.')