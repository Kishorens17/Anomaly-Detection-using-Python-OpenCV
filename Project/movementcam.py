import cv2
import numpy as np
from time import sleep

min_width = 80
min_height = 80  
offset = 6  
line_position = 550 
delay = 60  
detected = []  
person = 0  

def get_center(x, y, w, h):
    """Function to get the center point of a bounding box"""
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

subtractor = cv2.createBackgroundSubtractorMOG2() 

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera")
        break

    time = float(1 / delay)
    sleep(time)
    
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    
    img_sub = subtractor.apply(blur)
    
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, line_position), (1200, line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        center = get_center(x, y, w, h)
        detected.append(center)
        
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (cx, cy) in detected:
            if cy < (line_position + offset) and cy > (line_position - offset):
                person += 1
                cv2.line(frame1, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                detected.remove((cx, cy))
                print("Person detected: " + str(person))

    
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detect", dilated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
