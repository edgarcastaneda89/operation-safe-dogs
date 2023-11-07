import cv2
import numpy as np
from ultralytics import YOLO
import torch

def getBoxes():
    try:
        # Open the video file
        video1 = cv2.VideoCapture("dogsOne.mp4")
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

            #TODO  have to change the "Dogs is out to happen every second. Because if not will it spit out hundreds of strings when seeing one?"
            # Check if dogs are detected
            if cls == 16:
                dogDetected = True
                print("Dog is out!!")

        # Box coordinates (x, y, x2, y2) being printed
        print(bboxes)

        cv2.imshow("Img", frame)
        key = cv2.waitKey(0)

        # Prints classes
        print(classes)

        # Press Esc key to break the loop
        if key == 27:  # Use 27 for the Esc key (32 is for the Spacebar)
            break

if __name__ == "__main__":
    getBoxes()
