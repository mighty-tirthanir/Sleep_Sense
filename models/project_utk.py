import os
import cv2
import numpy as np
import csv
import warnings
import importlib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# --- Compatibility Fix for Mediapipe + Protobuf ---
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import google.protobuf.message_factory as mf
    if not hasattr(mf, "GetMessageClass"):
        # Patch to fix Mediapipe <-> Protobuf incompatibility
        mf.GetMessageClass = mf.GetMessages
except Exception:
    pass

# --- Import Mediapipe Safely ---
try:
    import mediapipe as mp
except ImportError:
    os.system("pip install mediapipe --quiet")
    import mediapipe as mp

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# ---------- Utility Functions ----------
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(eye):
    up = compute(eye[1], eye[5]) + compute(eye[2], eye[4])
    down = compute(eye[0], eye[3])
    return up / (2.0 * down)

def mouth_open_ratio(mouth):
    vertical = compute(mouth[2], mouth[3])
    horizontal = compute(mouth[0], mouth[1])
    return vertical / horizontal

# ---------- Color-Based Fatigue Detection ----------
def analyze_fatigue(frame, landmarks):
    LEFT_UNDER_EYE = [145, 153, 154, 155, 133, 173]
    RIGHT_UNDER_EYE = [374, 380, 381, 382, 263, 390]

    def extract_under_eye_region(points):
        pts = np.array([[int(landmarks[i][0]), int(landmarks[i][1])] for i in points])
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi, mask

    left_roi, left_mask = extract_under_eye_region(LEFT_UNDER_EYE)
    right_roi, right_mask = extract_under_eye_region(RIGHT_UNDER_EYE)

    lab_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2LAB)
    lab_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2LAB)

    L_left = cv2.mean(lab_left[:, :, 0], mask=left_mask)[0]
    L_right = cv2.mean(lab_right[:, :, 0], mask=right_mask)[0]
    L_avg = (L_left + L_right) / 2

    fatigue_score = max(0, min(100, 100 - L_avg))
    fatigue_level = "High Fatigue" if fatigue_score > 25 else "Low Fatigue"
    return fatigue_score, fatigue_level

# ---------- Detection ----------
def detect_drowsiness_fatigue_yawn(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])

            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            left_ratio = blinked(left_eye)
            right_ratio = blinked(right_eye)
            avg_EAR = (left_ratio + right_ratio) / 2

            INNER_MOUTH = [78, 308, 13, 14, 87, 317]
            mouth_pts = np.array([[int(landmarks[i][0]), int(landmarks[i][1])] for i in INNER_MOUTH])
            mouth_ratio = mouth_open_ratio(mouth_pts)
            yawn = "Yawning" if mouth_ratio > 0.6 else "Not Yawning"

            fatigue_score, fatigue_level = analyze_fatigue(frame, landmarks)

            if avg_EAR <= 0.23 or yawn == "Yawning" or fatigue_level == "High Fatigue":
                drowsy_state = "Drowsy"
            else:
                drowsy_state = "Not Drowsy"

            return [
                os.path.basename(image_path),
                drowsy_state,
                round(avg_EAR, 3),
                yawn,
                round(mouth_ratio, 3),
                fatigue_level,
                fatigue_score
            ]
    return [os.path.basename(image_path), "No Face", 0, "N/A", 0, "N/A", 0]

# ---------- Dataset Paths ----------
non_drowsy_path = r"D:\codes\project\models\datasets\train\alert"
drowsy_path = r"D:\codes\project\models\datasets\train\drowsy"
output_csv = "DDD_results.csv"

results = []
folders = [(non_drowsy_path, "Non Drowsy"), (drowsy_path, "Drowsy")]

for folder, true_label in folders:
    print(f"\nProcessing folder: {true_label}")
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, img_name)
            res = detect_drowsiness_fatigue_yawn(img_path)
            if res:
                res.append(true_label)
                results.append(res)
            print(res)

# ---------- Save Results ----------
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image Name", "Predicted Drowsiness", "Avg EAR", "Yawning", "Mouth Ratio",
        "Fatigue Level", "Fatigue Score", "True Label"
    ])
    writer.writerows(results)

# ---------- Evaluation ----------
y_true, y_pred = [], []
for row in results:
    if row[1] != "No Face":
        y_true.append("Drowsy" if row[7] == "Drowsy" else "Not Drowsy")
        y_pred.append(row[1])

if y_true:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Drowsy")
    rec = recall_score(y_true, y_pred, pos_label="Drowsy")
    f1 = f1_score(y_true, y_pred, pos_label="Drowsy")
    cm = confusion_matrix(y_true, y_pred, labels=["Drowsy", "Not Drowsy"])

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Drowsy", "Pred Not Drowsy"],
                yticklabels=["Actual Drowsy", "Actual Not Drowsy"])
    plt.title("Drowsiness Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
else:
    print("No valid faces detected for evaluation.")
