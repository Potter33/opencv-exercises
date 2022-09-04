"""
App that detect if a object contains the orange color
TODO: Search for a better orange range of colors
"""
import cv2
from object_detector import *
import numpy as np

# Load Object Detector
detector = HomogeneousBgDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Orange color min range
ORANGE_MIN = np.array([10, 100, 20],np.uint8)

# Orange color max range
ORANGE_MAX = np.array([15, 255, 255],np.uint8)

while True:
    ret, frame = cap.read()
    if ret == True:
        # Convert the frame to HSV
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create a mask for the orange color
        mask_orange = cv2.inRange(frameHSV, ORANGE_MIN, ORANGE_MAX)
        # Detect the orange object
        do = detector.detect_mask_object(frame, mask_orange, (0, 0, 255))
        # Show the frame
        cv2.imshow('Detect Orange Color', frame)
        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()