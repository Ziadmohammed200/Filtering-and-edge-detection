
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

def adaptive_threshold(image, block_size=11, C=2, method='mean', sigma=2):
    """
    Args:
        image: Grayscale image (uint8).
        block_size: Odd integer (e.g., 3, 5, 11).
        C: Constant subtracted from the mean.
        method: 'mean' or 'gaussian'.
        sigma: Standard deviation for Gaussian kernel.
    Returns:
        Binary thresholded image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if block_size % 2 == 0:
        block_size += 1
    k = block_size // 2
    
    # Pad image to handle borders
    padded = cv.copyMakeBorder(image, k, k, k, k, cv.BORDER_REFLECT)
    binary = np.zeros_like(image, dtype=np.uint8)
    
    if method == 'gaussian':
        x = np.arange(-k, k+1)
        y = np.arange(-k, k+1)
        x, y = np.meshgrid(x, y)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+block_size, j:j+block_size]
            
            if method == 'mean':
                threshold = np.mean(region) - C
            elif method == 'gaussian':
                threshold = np.sum(region * kernel) - C
            
            binary[i, j] = 255 if image[i, j] > threshold else 0
    
    return binary


