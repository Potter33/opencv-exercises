"""
App that using an Aruco maker detect a object and measures it
"""
import cv2
from object_detector import *
import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# Load Object Detector
detector = HomogeneousBgDetector()

# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if corners:

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(frame)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(frame, [box], True, (255, 0, 0), 2)
            cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    # Show the frame
    cv2.imshow("Measure object", frame)
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()