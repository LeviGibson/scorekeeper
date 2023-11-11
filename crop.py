import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage
from PIL import Image
from scipy.ndimage.filters import gaussian_filter


OUT_OF_FRAME_Y_AXIS = 1
OUT_OF_FRAME_X_AXIS = 2

def display(image : np.array):
    if image.shape[-1] == 3:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, cmap='gray')
        plt.show()

def find_lines(image):
    # image = 
    display(image)
    return image
    # bw = ((bw - np.min(bw)) * (1/np.max(bw)) * 255).astype('uint8')
    # # bw = 255-bw
    # bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    # # image = (image + bw).astype(int)
    # blurredBw = cv2.GaussianBlur(bw, (301, 301), 0).astype(float)
    # bw = (bw - blurredBw)
    # bw = ((bw - np.min(bw)) * (1/np.max(bw)) * 255).astype('uint8')

def filter_out_red_pen(image):
    bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
    bw = (image.astype(float)[:, :, 2] - bw)
    bw = np.stack((bw,)*3, axis=-1)
    # bw = gaussian_filter(bw, sigma=5)

    #Filters out red pen
    # image = image.astype(float)
    
    # image += (bw*4)
    
    blurred = cv2.GaussianBlur(image, (501, 501), 0)
    image[bw>15] = 0
    blurred *= (bw>15)
    image = image + blurred
    
    return image
    

def find_number(kernel : np.array, image : np.array):
    # image = filter_out_red_pen(image)
        
    if (image.shape[-1] == 3):
        kernel = cv2.cvtColor(kernel, cv2.COLOR_GRAY2RGB)
    
    method = cv2.TM_CCOEFF
    img = image.copy()
    result = cv2.matchTemplate(image, kernel, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc
        
    # display(image)
    # display(kernel)
    # cv2.imshow('image', image)
    # cv2.imshow('kernel', kernel)
    # cv2.waitKey(0)

def calculate_angle(a, b):
    return math.degrees(math.atan(b/a))

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

    
def crop(image: np.array):
    #Get locations of the Number 1 and the number 5 on the scorecard
    one = cv2.imread('numbers/1.png', 0)
    one = cv2.resize(one, (0, 0), fx=0.3, fy=0.3)
    oneloc = find_number(one, image)
    
    five = cv2.imread('numbers/5.png', 0)
    five = cv2.resize(five, (0, 0), fx=0.3, fy=0.3)
    fiveloc = find_number(five, image)
    
    #Calculate the angle at which the scorecard is rotated
    a = fiveloc[1] - oneloc[1]
    b = fiveloc[0] - oneloc[0]
    rotateAngle = calculate_angle(a, b)
    
    #Rotate the scorecard to be facing the normal direction
    pivot = oneloc
    # padX = [image.shape[1] - pivot[0], pivot[0]]
    # padY = [image.shape[0] - pivot[1], pivot[1]]
    # imgP = np.pad(image, [padY, padX], 'constant')
    
    image = ndimage.rotate(image, -rotateAngle, reshape=False)
    
    # image = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

    #Crop the scorecard
    c = round(math.sqrt(a**2 + b**2))
    imageCenter = (image.shape[1]//2, image.shape[0]//2)
    newOneLoc = rotate(imageCenter, oneloc, math.radians(rotateAngle))
    newOneLoc = (round(newOneLoc[0]), round(newOneLoc[1]))
    
    y1 = round(newOneLoc[1] - (c*.4))
    y2 = round((newOneLoc[1]+(c*1.16)))
    
    x1 = newOneLoc[0]
    x2 = round(newOneLoc[0]+(c*1.9))
    
    if y1 < 0 or y2 > image.shape[0]:
        return OUT_OF_FRAME_Y_AXIS, None
    
    if x2 > image.shape[1]:
        return OUT_OF_FRAME_X_AXIS, None
    
    image = image[y1:y2 , x1:x2]
    
    # display(image)
    return 0, image
    
    # b = base
    # h = height
    # c = hypotenuse
    
# one = cv2.imread('numbers/1.png', 0)
# img2 = cv2.imread('blank.jpg', 0)
# img2 = cv2.resize(img2, (0, 0), fx=0.3, fy=0.3)
# one = cv2.resize(one, (0, 0), fx=0.4, fy=0.4)


# display(crop(img2))
# print(find_number(one, img2))

# img1 = cv2.resize(img1, (0, 0), fx=0.35, fy=0.35)
# img2 = cv2.resize(img2, (0, 0), fx=0.2, fy=0.2)

# orb = cv2.ORB_create(nfeatures=1000)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# good=[]
# for m, n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
        
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)


# cv2.imshow('img3', img3)
# cv2.waitKey(0)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
