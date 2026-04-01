import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
import numpy as np

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_dark_circle(eye_landmarks, frame):
    x_min = np.min(eye_landmarks[:,0])
    x_max = np.max(eye_landmarks[:,0])
    y_min = np.max(eye_landmarks[:,1]) 
    height = int((x_max - x_min)/2)   
    y_max = min(y_min + height, frame.shape[0]-1)
    
    under_eye = frame[y_min:y_max, x_min:x_max]
    if under_eye.size == 0:
        return 255 
    
    gray_patch = cv2.cvtColor(under_eye, cv2.COLOR_BGR2GRAY)
    avg_intensity = gray_patch.mean()
    return avg_intensity


thresh = 0.23      
frame_check = 20    
flag = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        LeftEye = shape[lStart:lEnd]
        RightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(LeftEye)
        rightEar = eye_aspect_ratio(RightEye)
        ear = (leftEar + rightEar)/2.0

        cv2.drawContours(frame, [cv2.convexHull(LeftEye)], -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(RightEye)], -1, (0,255,0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "ALERT: Drowsy!", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        else:
            flag = 0

        left_dark = detect_dark_circle(LeftEye, frame)
        right_dark = detect_dark_circle(RightEye, frame)
        avg_dark = (left_dark + right_dark)/2.0

        if avg_dark < 100:
            sleep_statement = "Poor sleep quality: Dark circles detected."
        elif avg_dark < 140:
            sleep_statement = "Slight under-eye darkness: Rest more."
        else:
            sleep_statement = "Sleep quality looks normal."

        cv2.putText(frame, sleep_statement, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Sleep & Drowsiness Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
