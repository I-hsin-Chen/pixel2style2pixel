import numpy as np
from PIL import Image
import cv2

# img = Image.open('warping_sample/sample2/reconstructed.png')
# lr_img = np.array(img)

from ISR.models import RDN
import time

def super_resolution(path):
    img = Image.open(path)
    lr_img = np.array(img)
    rdn = RDN(weights='psnr-small')
    sr_img = rdn.predict(lr_img)
    Image.fromarray(sr_img)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, sr_img)
    cv2.imwrite('mid_result/after_super.png', sr_img)