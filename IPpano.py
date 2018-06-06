import numpy as np
import cv2
from matplotlib import pyplot as plt

L_ratio = 0.75
min_match = 4

img1 = cv2.imread("image1.png",-1)
img2 = cv2.imread("image2.jpg",-1)

img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(10000)

kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)

kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(des1,des2)

if matches is None:
    return None

matches = sorted(matches, key = lambda x:x.distance)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

if len(good) >= min_match :
  pts1 = np.float32([kp1[i] for (_, i) in good])
  pts2 = np.float32([kp2[i] for (i, _) in good])

(M, mask) = cv2.findHomography(pts1, pts2, cv2.RANSAC,min_match)

f_img = cv2.warpPerspective(img1, H,)

cv2.imshow('image',img1)
waitkey(0)
