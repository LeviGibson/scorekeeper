import cv2
import crop
import numpy as np
import math
from scipy import ndimage

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

def rotate_rect_around_point(rect, point):
    pass

def fix_rect_rotation(rects):
    for i, rect in enumerate(rects):
        rect = list(rect)
        if abs(abs(rect[2]) - 90) < abs(rect[2]):
            rect[1] = (rect[1][1], rect[1][0])
            if rect[2] < 0:
                rect[2] += 90
            else:
                rect[2] -= 90
                
        if abs(abs(rect[2]) - 90) < abs(rect[2]):
            rect[1] = (rect[1][1], rect[1][0])
            if rect[2] < 0:
                rect[2] += 90
            else:
                rect[2] -= 90
                
        rects[i] = rect
    return rects

def average_rect_rotation(rects):
    rot = 0
    for i in rects:
        rot += i[2]
    return rot/len(rects)

def rotate_image(image, rects):
    # cv2.drawContours(image, np.int0(cv2.boxPoints(rects)), -1, (0,255,0), 3)
    
    rects = fix_rect_rotation(rects)
    rotation = average_rect_rotation(rects)
    
    for i, r in enumerate(rects):
        r[0] = rotate((image.shape[1]/2, image.shape[0]/2), r[0], math.radians(-rotation))
        r[2] = 0
        rects[i] = r
    
    image = ndimage.rotate(image, rotation, reshape=False)
    
        
    return image, rects

def draw_rects(image, rects):
    for r in rects:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0,255,0), 3)
        
# def rect_bound_box        

def merge_duplicate_rects(rects):
    i = 0
    newRects = []
    entered = [False]*len(rects)
    
    for i1, r1 in enumerate(rects):
        if entered[i1]: continue
        shortlist = [r1]
        entered[i1] = True
        for i2, r2 in enumerate(rects):
            if i1 == i2: continue
            if entered[i2]: continue
            
            if abs(r1[0][1] - r2[0][1]) < 20:
                shortlist.append(r2)
                entered[i2] = True
        
        # maxSize = -1
        # maxRect = None
        # for element in shortlist:
        #     esize = element[1][0] * element[1][1]
        #     if esize > maxSize:
        #         maxRect = element
        #         maxSize = esize
        
        # newRects.append(maxRect)
        
        meanTop = 0
        meanBottom = 0
        minLeft = 100000
        maxRight = 0
        
        for element in shortlist:
            left = element[0][0] - (element[1][0]//2)
            right = element[0][0] + (element[1][0]//2)
            top = element[0][1] + (element[1][1]//2)
            bottom = element[0][1] - (element[1][1]//2)
            
            minLeft = min(left, minLeft)
            maxRight = max(right, maxRight)
            meanTop += top
            meanBottom += bottom
        
        meanTop //= len(shortlist)
        meanBottom //= len(shortlist)
        
        newRects.append((((minLeft + maxRight)//2, (meanTop+meanBottom)//2), (maxRight-minLeft, (meanBottom-meanTop)*1.1), 0))
        
    return newRects

def find_lines(image):
    # image = crop.filter_out_red_pen(image)
    
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # image = ((image[:, :, 1]>80)*255).astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    boxes = []
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 300:
            continue
        if(hierarchy[i][2] < 0 and hierarchy[i][3] < 0):
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        if (abs(cv2.contourArea(contour) - 
               cv2.contourArea(box)) / cv2.contourArea(contour)) > .15:
            continue
        boxes.append(rect)
        
    if len(boxes) == 0:
        print("No scorecard detected")
        return 1, None, None
    image, boxes = rotate_image(image, boxes)
    boxes = merge_duplicate_rects(boxes)
    
    if len(boxes) != 8:
        print("Incorrect Number of Boxes {}", len(boxes))
        return 1, None, None
    
    return 0, image, boxes

def crop_boxes(image, boxes):
    ret = []
    boxes = reversed(boxes)
    for box in boxes:
        left = int(box[0][0] - (box[1][0]//2))
        right = int(box[0][0] + (box[1][0]//2))
        top = int(box[0][1] + (box[1][1]//2))
        bottom = int(box[0][1] - (box[1][1]//2))
        time = image[top:bottom, left:right]
        ret.append(time)
        
    return ret
