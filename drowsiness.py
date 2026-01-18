import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading
import os

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_SOUND = os.path.join(BASE_DIR, "assets", "mixkit-rooster-crowing-in-the-morning-2462.wav")

# -------------------- MediaPipe Setup --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# -------------------- Eye Landmarks --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------- Thresholds --------------------
EAR_THRESHOLD = 0.21
DROWSY_TIME = 2.5  # seconds

# -------------------- Counters --------------------
blink_count = 0
eye_closed_start = None
alarm_on = False

# -------------------- Alarm Function --------------------
def play_alarm():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        playsound(ALARM_SOUND)
        alarm_on = False

# -------------------- EAR Calculation --------------------
def eye_aspect_ratio(eye_points, landmarks):
    p1 = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    p2 = np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    p3 = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])
    p4 = np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    p5 = np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y])
    p6 = np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])

    vertical = np.linalg.norm(p1 - p2) + np.linalg.norm(p3 - p4)
    horizontal = np.linalg.norm(p5 - p6)

    return vertical / (2.0 * horizontal)

# -------------------- Webcam --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape

            # =================  FACE BOX =================
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "FACE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # =================  EYE BOXES =================
            for eye, label in [(LEFT_EYE, "L-EYE"), (RIGHT_EYE, "R-EYE")]:
                ex = [int(landmarks[i].x * w) for i in eye]
                ey = [int(landmarks[i].y * h) for i in eye]

                cv2.rectangle(
                    frame,
                    (min(ex), min(ey)),
                    (max(ex), max(ey)),
                    (0, 255, 255),
                    2
                )

                cv2.putText(frame, label,
                            (min(ex), min(ey) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            # ================= EAR CALCULATION =================
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ================= DROWSINESS LOGIC =================
            if ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    elapsed = time.time() - eye_closed_start
                    if elapsed > DROWSY_TIME:
                        cv2.putText(frame, "DROWSY ALERT!", (100, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                        if not alarm_on:
                            threading.Thread(
                                target=play_alarm,
                                daemon=True
                            ).start()
            else:
                if eye_closed_start is not None:
                    blink_count += 1
                eye_closed_start = None
                alarm_on = False

            cv2.putText(frame, f"Blinks: {blink_count}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Eye Blink & Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
