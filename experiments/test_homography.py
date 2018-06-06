import cv2

import numpy as np

from tcp.registration.homography import Homography
from tcp.configs.alberta_config import Config

cnfg = Config()
hm = Homography(cnfg)

img = cv2.imread('Debug_Imgs/alberta_empty_intersection.png')
img_warped = hm.apply_homography_on_img(img)
