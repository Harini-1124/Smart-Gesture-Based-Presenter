import streamlit as st
import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# --- Streamlit Page Config ---
st.set_page_config(page_title="Smart Gesture Presenter", layout="wide")

# --- Parameters ---
width, height = 1920, 1080
gestureThreshold = 300
folderPath = "Presentation"

# --- Session State Initialization ---
if "imgNumber" not in st.session_state:
    st.session_state.imgNumber = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = [[]]
if "annotationNumber" not in st.session_state:
    st.session_state.annotationNumber = -1
if "annotationStart" not in st.session_state:
    st.session_state.annotationStart = False
if "zoom_scale" not in st.session_state:
    st.session_state.zoom_scale = 1.0
if "erase_mode" not in st.session_state:
    st.session_state.erase_mode = False

# --- Load Slides ---
if not os.path.exists(folderPath):
    st.error(f"⚠️ Folder '{folderPath}' not found. Please create it and add some slides.")
    st.stop()

pathImages = sorted(os.listdir(folderPath), key=len)
if not pathImages:
    st.error(f"⚠️ No images found in '{folderPath}'. Please add slide images.")
    st.stop()

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# --- Streamlit Layout ---
st.title("🖐️ Smart Gesture Based Presenter")
col1, col2 = st.columns(2)

start_button = st.button("▶ Start Presenter")
stop_button = st.button("⏹ Stop Presenter")

if start_button:
    run = True
elif stop_button:
    run = False
else:
    run = False

if run:
    frame_window1 = col1.empty()
    frame_window2 = col2.empty()

    while True:
        success, img = cap.read()
        if not success:
            st.error("⚠️ Camera not detected!")
            break

        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[st.session_state.imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        # Hand Detection
        hands, img = detectorHand.findHands(img)
        cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

        if hands:
            hand = hands[0]
            cx, cy = hand["center"]
            lmList = hand["lmList"]
            fingers = detectorHand.fingersUp(hand)

            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
            indexFinger = xVal, yVal

            # Slide Navigation
            if cy <= gestureThreshold:
                if fingers == [1, 0, 0, 0, 0]:
                    if st.session_state.imgNumber > 0:
                        st.session_state.imgNumber -= 1
                        st.session_state.annotations = [[]]
                        st.session_state.annotationNumber = -1
                        st.session_state.annotationStart = False
                if fingers == [0, 0, 0, 0, 1]:
                    if st.session_state.imgNumber < len(pathImages) - 1:
                        st.session_state.imgNumber += 1
                        st.session_state.annotations = [[]]
                        st.session_state.annotationNumber = -1
                        st.session_state.annotationStart = False

            # Pointer
            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            # Draw
            if fingers == [0, 1, 0, 0, 0]:
                if not st.session_state.annotationStart:
                    st.session_state.annotationStart = True
                    st.session_state.annotationNumber += 1
                    st.session_state.annotations.append([])
                st.session_state.annotations[st.session_state.annotationNumber].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            else:
                st.session_state.annotationStart = False

            # Erase
            if fingers == [0, 1, 1, 1, 0]:
                if st.session_state.annotations:
                    st.session_state.annotations.pop(-1)
                    st.session_state.annotationNumber -= 1

            # Zoom
            if fingers == [1, 1, 1, 1, 1]:
                st.session_state.zoom_scale += 0.1
            if fingers == [0, 0, 0, 0, 0]:
                st.session_state.zoom_scale = max(1.0, st.session_state.zoom_scale - 0.1)

            # Erase mode
            if fingers == [1, 1, 0, 0, 0]:
                st.session_state.erase_mode = True
            else:
                st.session_state.erase_mode = False

        # Draw annotations
        for annotation in st.session_state.annotations:
            for j in range(1, len(annotation)):
                color = (255, 255, 255) if st.session_state.erase_mode else (0, 0, 200)
                thickness = 50 if st.session_state.erase_mode else 12
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], color, thickness)

        # Zoom effect
        zoomed_img = cv2.resize(imgCurrent, None, fx=st.session_state.zoom_scale, fy=st.session_state.zoom_scale)
        center_x, center_y = width // 2, height // 2
        try:
            cropped_img = zoomed_img[
                int(center_y - height // 2): int(center_y + height // 2),
                int(center_x - width // 2): int(center_x + width // 2)
            ]
        except:
            cropped_img = imgCurrent

        # Show in Streamlit
        frame_window1.image(cropped_img, channels="BGR")
        frame_window2.image(img, channels="BGR")

        # Stop loop if button clicked
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
