import numpy as np
import cv2
from matplotlib import pyplot as plt

L_ratio = 0.75
min_match = 4
n=6

def orb_feat(image):

  orb = cv2.ORB_create()
  orb.setMaxFeatures(50000)

  kp1, des1 = orb.detectAndCompute(image, None)

 # kp1 = np.float32([kp.pt for kp in kp1])

  return (kp1,des1)


def feat_match_Homo(img1,img2,kp1,kp2,des1,des2,L_ratio,min_match):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1,des2,k=2)

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

def warping(imageA,imageB):
    result = cv2.warpPerspective(imageA, M,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result


def GC(imageA,imageB):
    A = cv2.warpPerspective(imageA, M,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
    cv2.imshow('av',A)
    for i in range(imageB.shape[0]-1):
        for j in range(imageB.shape[1]-1):
            if A[i,j]*imageB[i,j]>0:
                mean = int((A[i,j]+imageB[i,j])/2)
                A[i,j] = mean
                imageB[i,j] = mean
    A[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    cv2.imshow('a',A)
    cv2.waitKey(0)

def blend(imageA,imageB):
# generate Gaussian pyramid for A
    G = imageA.copy()
    gpA = [G]
    for i in range(n):
        G = cv2.pyrDown(G)
        gpA.append(G)

# generate Gaussian pyramid for B
    G = imageB.copy()
    gpB = [G]
    for i in range(n):
        G = cv2.pyrDown(G)
        gpB.append(G)

# generate Laplacian Pyramid for A
    lpA = [gpA[n-1]]
    for i in range(n-1,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize = size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

# generate Laplacian Pyramid for B
    lpB = [gpB[n-1]]
    for i in range(n-1,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize = size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

# Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols = la.shape
        ls = warping(la,lb)
        #ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
        LS.append(ls)

# now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1]+LS[i].shape[1]%2, LS[i].shape[0]+LS[i].shape[0]%2)
        ls_ = cv2.pyrUp(ls_,dstsize = size)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

img2 = cv2.imread("two.jpg",0)
img1 = cv2.imread("three.jpg",0)
imageA = img1
imageB = img2
#imageA = cv2.resize(imageA,(500,500))
#imageB = cv2.resize(imageB,(500,500))
(kp1,des1) = orb_feat(img1)
(kp2,des2) = orb_feat(img2)
print(len(kp1),"and",len(des1),"so")
(A,M,c) = feat_match_Homo(img1,img2,kp1,kp2,des1,des2,L_ratio,min_match)
#im1,im2 = gain_comp(img1,img2)
result = warping(imageA,imageB)
result = cv2.resize(result,(1400,1400))
#result1 = warping(im1,im2)
x = GC(imageA,imageB)
#result1 = blend(imageA,imageB)
#result1 = cv2.resize(result,(1400,1400))
#print(result[int(imageB.shape[0]/2),int(imageB.shape[1]/2)])
#result = stitching(img1,img2,b)
#img2 = cv2.drawKeypoints(img1,kp,None,color=(0,255,0),flags=0)
#plt.imshow(img2),plt.show()
#print ("I have", len(kp))
#result = gain_com(result,imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0],imageB.shape[1],imageB.shape[0])
cv2.imshow('image2',result)
cv2.waitKey(0)
cv2.imshow('image',result1)
cv2.waitKey(0)
#cv2.destroyAllWindows()
