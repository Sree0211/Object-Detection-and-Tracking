import cv2

# Load the input traffic video
cap = cv2.VideoCapture("/Users/sreenathswaminathan/Desktop/Uni-Docs/Autonomous fahren kurs/CV Project/Object-Detection-and-Tracking/pexels_videos_1171461 (1080p).mp4")


# Mask
# Contours
# Object detectors
# ROI

# Object detection using our stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=45)

while cap.isOpened():

    ref, frame = cap.read() 

    height, width, _ = frame.shape

    # Region of interest to count the vehicle
    roi = frame[200:1000,15:800]
    
    # Mask to distinguish constant and moving objects in the video
    mask = object_detector.apply(roi)
    _ , mask = cv2.threshold(mask,253,255,cv2.THRESH_BINARY)
    
    # Create contours to sketch the objects in the frame
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Get the area of each contour
        if area > 100:
            cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w,y+h),(0,255,0),3)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) == 27:
        break

print(height,width)
cap.release()
cv2.destroyAllWindows()