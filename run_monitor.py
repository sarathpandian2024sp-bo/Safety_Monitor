import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from fire_datasets import canonical_label, load_fire_models, merge_boxes, model_message

# ==========================================
# 1. CONFIGURATION (EDIT ONLY THIS SECTION)
# ==========================================
CAMERA_SOURCE = 0             # 0 = PC Webcam | "http://192.168.1.15:8080/video" = Mobile App | "rtsp://..." = CCTV
CONFIDENCE_THRESHOLD = 0.50   # AI certainty threshold (0.50 = 50%)
CONFIRMATION_FRAMES = 4       # Consecutive processed frames needed to trigger alarm
ALARM_HOLD_SECONDS = 4        # Minimum seconds the speaker alarm stays active

# Create incident log folder automatically
os.makedirs("incident_logs", exist_ok=True)

# ==========================================
# 2. BUILT-IN ALARM SOUND GENERATOR
# ==========================================
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def create_tone(freq_hz, duration_sec=0.4):
    """Generates a pure audio tone using NumPy so external .wav files are not needed."""
    sample_rate = 44100
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    wave = np.sin(freq_hz * t * 2 * np.pi)
    audio = (wave * 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    if channels > 1:
        audio = np.repeat(audio[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(audio)

fire_siren_tone = create_tone(1200, 0.4)   # High-pitch urgent tone (1200 Hz) for Fire/Smoke
worker_sign_tone = create_tone(650, 0.4)   # Lower-pitch tone (650 Hz) for Worker Hand Sign

# ==========================================
# 3. LOAD AI MODELS (YOLO + MEDIAPIPE)
# ==========================================
print("Loading D-Fire (Gaia) and the local fire/smoke set...")
fire_models = load_fire_models(
    hf_hub_download, YOLO, os.path.dirname(os.path.abspath(__file__))
)
print(model_message(fire_models))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.70, min_tracking_confidence=0.70)

# ==========================================
# 4. HAND-SIGN & ALARM HELPER FUNCTIONS
# ==========================================
def is_open_palm_sign(hand_landmarks):
    """
    Open-palm worker distress sign:
    all four fingertips (8, 12, 16, 20) sit higher than their middle knuckles (6, 10, 14, 18).
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(finger_tips, finger_pips)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
    )
    return extended >= 4

fire_counter = 0
sign_counter = 0
last_alert_time = 0
current_alarm = None

def update_alarm_state(detected_type, frame):
    global last_alert_time, current_alarm
    now = time.time()

    if detected_type:
        last_alert_time = now
        if current_alarm != detected_type:
            pygame.mixer.stop()
            if detected_type == "FIRE_SMOKE":
                fire_siren_tone.play(loops=-1)
            elif detected_type == "HAND_SIGN":
                worker_sign_tone.play(loops=-1)

            current_alarm = detected_type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"incident_logs/{detected_type}_{timestamp}.jpg"
            cv2.imwrite(log_path, frame)
            print(f"[ALERT TRIGGERED] {detected_type} -> Snapshot saved to {log_path}")
    else:
        if current_alarm and (now - last_alert_time > ALARM_HOLD_SECONDS):
            pygame.mixer.stop()
            print(f"[CLEAR] {current_alarm} resolved. Speaker muted.")
            current_alarm = None

# ==========================================
# 5. MAIN I5-OPTIMIZED VIDEO MONITORING LOOP
# ==========================================
cap = cv2.VideoCapture(CAMERA_SOURCE)
frame_count = 0
print("Starting i5-Optimized AI Safety Monitor... Press 'Q' on the video window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Waiting for camera stream...")
        time.sleep(1)
        continue

    # Resize to 640x480 for fast Core i5 CPU processing
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Skip every alternate frame to keep CPU load low
    if frame_count % 2 != 0:
        cv2.imshow("AI Safety Monitor (Press Q to Quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    fire_found = False
    sign_found = False

    # --- A. D-Fire and the local set. Either model can mark fire or smoke. ---
    found = []
    for source in fire_models:
        results = source["model"](frame, imgsz=416, conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                label = canonical_label(source["model"].names[int(box.cls)])
                if label not in ("fire", "smoke"):
                    continue
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                found.append({"xyxy": (x1, y1, x2, y2), "label": label, "conf": conf})
    for box in merge_boxes(found):
        fire_found = True
        x1, y1, x2, y2 = box["xyxy"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame, f"{box['label'].upper()} ({box['conf']:.2f})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )

    # --- B. Run Worker Open-Palm Hand-Sign Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            if is_open_palm_sign(hand_lms):
                sign_found = True
                cv2.putText(frame, "SIGN DETECTED: WORKER HELP", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # --- C. Confirm Over Multiple Frames & Trigger PC Speakers ---
    fire_counter = fire_counter + 1 if fire_found else 0
    sign_counter = sign_counter + 1 if sign_found else 0

    if fire_counter >= CONFIRMATION_FRAMES:
        update_alarm_state("FIRE_SMOKE", frame)
        cv2.putText(frame, "EMERGENCY: FIRE / SMOKE!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif sign_counter >= CONFIRMATION_FRAMES:
        update_alarm_state("HAND_SIGN", frame)
        cv2.putText(frame, "ALERT: WORKER ASSISTANCE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    else:
        update_alarm_state(None, frame)

    cv2.imshow("AI Safety Monitor (Press Q to Quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pygame.mixer.stop()
cap.release()
cv2.destroyAllWindows()
