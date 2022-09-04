import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    def detect_mask_object(self, frame, mask, color):
        """
        It takes a frame, a mask, and a color, and returns the contour of the object
        Args:
            frame: The frame to be processed
            mask: The mask to be used
            color: The color who will be used to draw the contour
        Returns:
            The contour of the object
        """
        try:
            # Find the contours of the frame
            cnts, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the index of the largest contour
            c = max(cnts, key=cv2.contourArea)
            # Get the moments of the largest contour
            x, y, w, h = cv2.boundingRect(c)
            # Draw the contour of the object
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Its Orange!", (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

            return (round(x + w / 2), round(y + h / 2))
        except:
            return None