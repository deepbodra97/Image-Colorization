import cv2
import numpy as np

def rgb_to_lab(image_bgr):
	return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB) / 255.	

def lab_to_rgb(image_lab):
	return cv2.cvtColor(np.uint8(image_lab*255.), cv2.COLOR_LAB2BGR)