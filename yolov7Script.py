import cv2
import numpy as np
from ultralytics import YOLO
import torch

#TODO create a function that tries to open the camera

def getBoxes():
    try:
        # Open the video file
        # RTSP stream URL
        rtsp_url = 'rtsp://admin:Telo1205@192.168.0.91:554/cam/realmonitor?channel=1&subtype=0'

        # Create a VideoCapture object for the RTSP stream
        video1 = cv2.VideoCapture(rtsp_url)



        model = YOLO("yolov8m.pt")
    except: 
        print("Video source not found!")

    while True:
        ret, frame = video1.read()
        if not ret:
            break

        results = model(frame, device="mps")
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")

        dogDetected = False


        for cls, bbox in zip(classes, bboxes):
            x, y, x2, y2 = bbox

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

            # Check if dogs are detected
            if cls == 16:
                dogDetected = True
                print("Dog is out!!")

        # Box coordinates (x, y, x2, y2) being printed
        print(bboxes)

        # Adjust frames 
        cv2.imshow("Img", frame)
        key = cv2.waitKey(20)

        # Prints classes
        print(classes)

        # Press Esc key to break the loop
        if key == 27:  # Use 27 for the Esc key (32 is for the Spacebar)
            break

if __name__ == "__main__":
    getBoxes()
