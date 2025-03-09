import cv2
import winsound
import numpy as np

cap = cv2.VideoCapture(0)
back_sub = cv2.createBackgroundSubtractorMOG2()
heavy_movement_threshold = 2000

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C://Users//kisho//OneDrive//Desktop//Amrita//SEM 3//Computer Networks//project//output.avi', fourcc, 20.0, (1280, 480))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    fg_mask = back_sub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heavy_movement_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > heavy_movement_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            heavy_movement_detected = True

    if heavy_movement_detected:
        cv2.putText(frame, "ALERT: Movement Detected!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        winsound.Beep(1000, 500)
        #winsound.Beep(2000, 500)
        #winsound.Beep(500, 500)

    fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, fg_mask_bgr))

    out.write(combined)
    
    cv2.imshow('Combined Output', combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()   
cv2.destroyAllWindows()