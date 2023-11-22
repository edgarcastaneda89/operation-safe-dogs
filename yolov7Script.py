import cv2
import numpy as np
from ultralytics import YOLO
import time
import http.client
import urllib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Pushover API credentials
pushoverApiToken = 'abqz3wn8oacd2r4up4a71nat7bbj1y'
pushoverUserKey = 'uhdb4kvk8egzh4nhr9w8iz77gyyjq3'

MAX_RETRIES = 3
dogCount = 0

# Push notification onto phone
def pushNotification(message):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
                 urllib.parse.urlencode({
                     "token": pushoverApiToken,
                     "user": pushoverUserKey,
                     "message": message,
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    response = conn.getresponse()
    print(response.read().decode())
    conn.close()

# One time connect to camera 
def connectToCamera(rtspUrl):
    return cv2.VideoCapture(rtspUrl)

# Reconnects to camera in case of unexpected shutdown
def reconnectToCamera(rtspUrl):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            videoCapture = connectToCamera(rtspUrl)
            print("Reconnected to the camera.")
            return videoCapture
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
            time.sleep(3)  # Add a delay before retrying

    print("Failed to reconnect after multiple attempts.")
    return None

# Detects objects I.E dogs
def processObjects():
    global dogCount  # Declare global variable

    rtspUrl = 'rtsp://admin:Telo1205@192.168.0.91:554/cam/realmonitor?channel=1&subtype=0'
    video1 = connectToCamera(rtspUrl)

    if video1 is None:
        print("Video source not found!")
        return

    model = YOLO("yolov8m.pt")

    while True:
        ret, frame = video1.read()
        if not ret:
            video1 = reconnectToCamera(rtspUrl)
            if video1 is None:
                break
            continue

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
                dogCount += 1  # Increment the dog count
                pushNotification("Dogs seen on camera!!")

        # Box coordinates (x, y, x2, y2) being printed
        print(bboxes)

        cv2.imshow("Img", frame)
        key = cv2.waitKey(20)

        # Prints classes
        print(classes)

        # Press Esc key to break the loop
        if key == 27 or dogCount >= 5:
            print("Dog count at 5.")
            break

    print("Program ended after detecting 5 dogs.")

if __name__ == "__main__":
    processObjects()