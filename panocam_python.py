import numpy as np
import cv2
from matplotlib import pyplot as plt

L_ratio = 0.75
min_match = 4


def orb_feat(image):

  orb = cv2.ORB_create()
  orb.setMaxFeatures(50000)

  kp1, des1 = orb.detectAndCompute(image, None)

 # kp1 = np.float32([kp.pt for kp in kp1])

  return (kp1,des1)


def feat_match_Homo(img1,img2,kp1,kp2,des1,des2,L_ratio,min_match):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.match(des1,des2)

    if matches is None:
        return None

#    matches = sorted(matches, key = lambda x:x.distance)

    good = []
    for m,n in matches:
        if m.distance < L_ratio*n.distance:
            good.append(m)

    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10000],None,flags=2)
    if len(good) >= min_match :
        #pts1 = np.float32([kp1[i] for (_, i) in good])
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #pts2 = np.float32([kp2[i] for (i, _) in good])
#def Warping(kp1,kp2,des1,des2,L_ratio,min_match):
        M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,min_match)
        #(A,B)=src_pts[0]
        #print(A)
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good[:1000],None,flags=2)
        #plt.imshow(img3),plt.show()
        return (good,M,mask)

    return None

#def stitching(imageA,imageB,M):

    #result = cv2.warpPerspective(imageA, b,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
    #result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB


img2 = cv2.imread("two.jpg",0)
img1 = cv2.imread("three.jpg",0)
imageA = img1
imageB = img2
imageA = cv2.resize(imageA,(500,500))
imageB = cv2.resize(imageB,(500,500))
(kp1,des1) = orb_feat(img1)
(kp2,des2) = orb_feat(img2)
print(len(kp1),"and",len(des1),"so")
(A,M,c) = feat_match_Homo(img1,img2,kp1,kp2,des1,des2,L_ratio,min_match)

result = cv2.warpPerspective(imageA, M,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

#result = stitching(img1,img2,b)
#img2 = cv2.drawKeypoints(img1,kp,None,color=(0,255,0),flags=0)
#plt.imshow(img2),plt.show()
#print ("I have", len(kp))

cv2.imshow('image',result)
cv2.waitKey(0)
#cv2.destroyAllWindows()
