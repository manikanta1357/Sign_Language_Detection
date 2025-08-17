import cv2
import numpy as np
import os

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# Ensure directories exist
train_dir = "C:\\Users\\raman\\OneDrive\\Documents\\OneDrive\\Desktop\\backend\\sign_language_detection\\code\\gesture\\train"
test_dir = "C:\\Users\\raman\\OneDrive\\Documents\\OneDrive\\Desktop\\backend\\sign_language_detection\\code\\gesture\\test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subdirectories for each class
for i in range(1, 11):
    os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(i)), exist_ok=True)

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
num_frames = 0
element = 1  # Start with class 1
num_imgs_taken = 0

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    elif num_frames <= 300:
        hand = segment_hand(gray_frame)
        cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.putText(frame_copy, str(num_frames) + " For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Thresholded Hand Image", thresholded)
            if num_imgs_taken < 200:
                img_name = f"{train_dir}/{element}/gesture_{num_imgs_taken}.jpg"
                cv2.imwrite(img_name, thresholded)
                print(f"Saved {img_name}")
                num_imgs_taken += 1
    else:
        hand = segment_hand(gray_frame)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            if num_imgs_taken < 400:
                img_name = f"{test_dir}/{element}/gesture_{num_imgs_taken - 200}.jpg"
                cv2.imwrite(img_name, thresholded)
                print(f"Saved {img_name}")
                num_imgs_taken += 1

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (0, 255, 0), 5)
    num_frames += 1
    cv2.imshow("Sign Detection", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or num_imgs_taken >= 400:
        if num_imgs_taken >= 400:
            element += 1
            num_imgs_taken = 0
            num_frames = 0
            if element > 10:
                break
        else:
            break

cam.release()
cv2.destroyAllWindows()