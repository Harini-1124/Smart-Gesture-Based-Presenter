from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
zoom_scale = 1.0
zoom_increment = 0.1
erase_mode = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]
        fingers = detectorHand.fingersUp(hand)

        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            if fingers == [1, 0, 0, 0, 0]:
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        if fingers == [1, 1, 1, 1, 1]:  # All fingers up -> Zoom in
            zoom_scale += zoom_increment
            buttonPressed = True

        if fingers == [0, 0, 0, 0, 0]:  # Fist -> Zoom out
            zoom_scale = max(1.0, zoom_scale - zoom_increment)
            buttonPressed = True

        if fingers == [1, 1, 0, 0, 0]:  # Eraser mode
            erase_mode = True
            buttonPressed = True

        if fingers == [1, 0, 1, 0, 1]:  # Custom gesture for slide number input (Thumb, Middle, Pinky)
            slide_input = input("Enter slide number: ")
            try:
                slide_number = int(slide_input)
                if 1 <= slide_number <= len(pathImages):
                    imgNumber = slide_number - 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            except ValueError:
                print("Invalid slide number")

    else:
        annotationStart = False
        erase_mode = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                if erase_mode:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (255, 255, 255), 50)
                else:
                    cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    # Apply zoom effect
    zoomed_img = cv2.resize(imgCurrent, None, fx=zoom_scale, fy=zoom_scale)
    center_x, center_y = width // 2, height // 2
    cropped_img = zoomed_img[
        int(center_y - height // 2): int(center_y + height // 2),
        int(center_x - width // 2): int(center_x + width // 2)
    ] if zoom_scale > 1 else zoomed_img

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = cropped_img.shape
    cropped_img[0:hs, w - ws: w] = imgSmall

    cv2.imshow("Slides", cropped_img)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break