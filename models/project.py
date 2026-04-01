import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[3], eye[0])
    ear = (A+B) / (2.0 *C)
    return ear
thresh = 0.23
flag = 0
frame_check = 20
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width =800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        LeftEye = shape[lStart: lEnd]
        RightEye = shape[rStart: rEnd]
        leftEar = eye_aspect_ratio(LeftEye)
        rightEar = eye_aspect_ratio(RightEye)
        ear = (leftEar + rightEar)/ 2.0
        lefteyeHull =  cv2.convexHull(LeftEye)
        righteyeHull =  cv2.convexHull(RightEye)
        cv2.drawContours(frame, [lefteyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [righteyeHull], -1, (0,255,0), 1)

        if ear< thresh:
            flag +=1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "ALERT", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
        else:
            flag = 0 



    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()