import mediapipe as mp
import math
import cv2
import time
import winsound


mp_draw = mp.solutions.drawing_utils
mp_mesh = mp.solutions.face_mesh


#  EAR
def calculate_EAR(landmarks, idx):
    p1 = landmarks[idx[0]]
    p2 = landmarks[idx[1]]
    p3 = landmarks[idx[2]]
    p4 = landmarks[idx[3]]
    p5 = landmarks[idx[4]]
    p6 = landmarks[idx[5]]

    A = math.dist((p2.x, p2.y), (p6.x, p6.y))
    B = math.dist((p3.x, p3.y), (p5.x, p5.y))
    C = math.dist((p1.x, p1.y), (p4.x, p4.y))

    EAR = (A + B) / (2.0 * C)
    return EAR

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# Head Turn Angle

def head_turn_angle(landmarks, w, h):
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]

    nose_x = nose.x * w
    left_x = left.x * w
    right_x = right.x * w

    face_center = (left_x + right_x) / 2
    angle = (nose_x - face_center)

    return angle

cap = cv2.VideoCapture(0)


#   METRICS FOR SCORING

start_time = time.time()
total_distracted_seconds = 0
blink_count = 0
eye_closed_last_frame = False

with mp_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    closed_frames = 0
    distracted_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                landmarks = face.landmark

                # EAR
                left_EAR = calculate_EAR(landmarks, LEFT_EYE)
                right_EAR = calculate_EAR(landmarks, RIGHT_EYE)
                EAR = (left_EAR + right_EAR) / 2

                cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # DROWSINESS
                if EAR < 0.21:
                    closed_frames += 1
                    if not eye_closed_last_frame:
                        blink_count += 1
                        eye_closed_last_frame = True
                else:
                    closed_frames = 0
                    eye_closed_last_frame = False

                if closed_frames > 15:
                    cv2.putText(frame, "DROWSY!", (50, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if closed_frames == 16:
                    winsound.MessageBeep()

                # HEAD TURN
                angle = head_turn_angle(landmarks, w, h)

                if abs(angle) > 45:
                    distracted_frames += 1
                else:
                    distracted_frames = 0

                # If distracted > 3 sec visually on 25 FpS
                if distracted_frames > 75:
                    cv2.putText(frame, "DISTRACTED!", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                if distracted_frames == 76:
                    winsound.MessageBeep()

                # Count distracted seconds for final score
                if distracted_frames > 75:
                    total_distracted_seconds += 1/25  # on 25 FPS


                mp_draw.draw_landmarks(
                    frame, face,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION
                )

        cv2.imshow("Driver Monitor", frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#  SCORE

total_time = time.time() - start_time

score = 100
score -= total_distracted_seconds * 1.2
score -= blink_count * 0.5
score = max(0, min(100, round(score)))
score = score / 100

# Rating
if score >= 0.8:
    rating = "SAFE"
elif score >= 0.6:
    rating = "MODERATE"
else:
    rating = "DANGEROUS"



print("\n------ DRIVER MONITORING SUMMARY ------")
print(f"Total driving time: {int(total_time)} sec")
print(f"Distracted time: {int(total_distracted_seconds)} sec")
print(f"Blink count: {blink_count}")
print(f"Final Focus Score: {score}")
print(f"Rating: {rating}")
print("---------------------------------------\n")
