import cv2
import numpy as np

# cv2.imshow('image_bgr', image_bgr)
# l, a, b = cv2.split(image_lab)
# cv2.merge((l, a, b))
# cv2.waitKey(0)
# cv2.imread('./data/face_images/image00000.jpg')
def rgb_to_lab(image_bgr):
	return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB) / 255.	

def lab_to_rgb(image_lab): # l, a, b should be scaled in [0, 1]
	return cv2.cvtColor(np.uint8(image_lab*255.), cv2.COLOR_LAB2BGR)