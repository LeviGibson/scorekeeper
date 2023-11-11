from crop import display
import cv2

def crop_to_time(image):
    s = image.shape
    return image[:, int(s[1]*.14):int(s[1]*.72)]

def read_time(image):
    image = crop_to_time(image)
    cv2.imwrite("thing.png", image)
    display(image)
