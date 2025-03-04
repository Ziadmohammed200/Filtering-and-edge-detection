
import cv2 as cv
import numpy as np

#Otsu's Binarization
def otsu_binarization(img):
    # make the img grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    blur = cv.GaussianBlur(img,(5,5),0)

    hist = cv.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) 
        q1,q2 = Q[i],Q[255]-Q[i]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i])

        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    _, binary_img = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY)
    return binary_img

#Adaptive (local) Thresholding
def local_thresholding(img):
    pass



