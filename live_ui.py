"""
Local CCTV dashboard.

Live video opens in the browser. Fire uses a 1200 Hz tone and worker assist
uses a 650 Hz tone. Both play on this PC for 30 seconds.
"""

import argparse
import base64
import hmac
import math
import os
import queue
import tempfile
import threading
import time
import wave
import winsound
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from fire_datasets import canonical_label, load_fire_models, merge_boxes, model_message

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "incident_logs")
os.makedirs(LOG_DIR, exist_ok=True)

FIRE_CONFIDENCE = 0.28
SMOKE_CONFIDENCE = 0.32
CONFIRMATION_FRAMES = 4
ALARM_HOLD_SECONDS = 30
DELETE_KEY = "****"
HOLD_FINGERS_SECONDS = 5.0
OPEN_CLOSE_CYCLES = 3
CYCLE_WINDOW_SECONDS = 12.0
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGER_COLORS = (
    (0, 220, 255),
    (80, 220, 90),
    (255, 190, 40),
    (220, 90, 255),
    (40, 90, 255),
)
PALM_LINKS = ((0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17))

try:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
except ImportError:
    hf_hub_download = None
    YOLO = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

TONE_DIR = os.path.join(tempfile.gettempdir(), "ai_safety_monitor_tones")
os.makedirs(TONE_DIR, exist_ok=True)


def _save_wav(path, samples, sample_rate=44100):
    audio = np.clip(samples, -32767, 32767).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def write_wav(path, freq_hz, volume, duration_sec=0.4):
    """Write the original steady tone. Volume is baked into the wave."""
    sample_rate = 44100
    count = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, count, False)
    samples = np.sin(freq_hz * t * 2 * np.pi) * (32767 * volume)
    _save_wav(path, samples, sample_rate)


def parse_source(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


class SafetyMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.camera_source = 0
        self.want_camera = False
        self.jpeg = None
        self.camera_open = False
        self.camera_message = "Press Start to open the live view."
        self.models_ready = False
        self.model_message = "D-Fire and the local fire/smoke set are loading..."
        self.hand_ready = False
        self.hand_message = "Hand-sign model is loading..."
        self.speaker_enabled = True
        self.volume = 0.85
        self.speaker_error = None
        self.current_alarm = None
        self.speaker_playing = False
        self.fire_counter = 0
        self.sign_counter = 0
        self.finger_count = 0
        self.hand_cycles = 0
        self.hand_hold = 0.0
        self.last_alert_time = 0
        self.boxes = []
        self.hand_points = []
        self.detections = []
        self.hand_visible = False
        self.preview_note = ""
        self.fps = 0.0
        self.models_loaded = False

        self._cap = None
        self._raw = None
        self._raw_seq = 0
        self._reopen = False
        self._model = None
        self._models = []
        self._hands = None
        self._hands_still = None
        self._audio_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._jobs = queue.Queue()
        self._smooth_points = None
        self._gesture_open = False
        self._open_frames = 0
        self._closed_frames = 0
        self._hold_started = None
        self._hold_grace_until = 0.0
        self._cycle_times = []
        self._last_hand_seen = 0.0
        self._last_points_time = 0.0
        self._count_hist = []
        self._shown_fingers = 0
        self._pending_boxes = []
        self._last_fire_boxes = []
        self._last_yolo = 0.0
        self._fire_misses = 0
        self._alarm_until = 0.0
        self._fire_path = os.path.join(TONE_DIR, "fire.wav")
        self._sign_path = os.path.join(TONE_DIR, "hand.wav")
        self._test_path = os.path.join(TONE_DIR, "test.wav")
        self._test_until = 0.0

        self._write_tones()
        self._publish(self._blank("START LIVE VIEW"))

    def _write_tones(self):
        try:
            level = max(0.0, min(1.0, self.volume))
            write_wav(self._fire_path, 1200, level, 0.4)
            write_wav(self._sign_path, 650, level, 0.4)
            write_wav(self._test_path, 880, level, 0.6)
            self.speaker_error = None
        except Exception as exc:
            self.speaker_error = f"Speaker init failed: {exc}"

    def _blank(self, text):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (18, 22, 28)
        cv2.putText(
            frame, text, (36, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 220, 230), 2, cv2.LINE_AA
        )
        return frame

    def _publish(self, frame):
        view = frame.copy()
        self._draw_overlay(view)
        ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return
        with self.lock:
            self.jpeg = buf.tobytes()

    def _paint(self, frame, boxes, points, alarm, note, live):
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            color = (40, 40, 255) if box["label"] == "fire" else (255, 210, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"{box['label'].upper()} ({box['conf']:.2f})",
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if len(points) >= 21:
            for start, end in PALM_LINKS:
                cv2.line(frame, points[start], points[end], (0, 190, 220), 2)
            for chain, color in zip(FINGER_CHAINS, FINGER_COLORS):
                for start, end in zip(chain, chain[1:]):
                    cv2.line(frame, points[start], points[end], color, 2)
                cv2.circle(frame, points[chain[-1]], 6, color, -1)
                cv2.circle(frame, points[chain[0]], 3, color, -1)
            cv2.circle(frame, points[0], 5, (0, 190, 220), -1)

        if live:
            cv2.circle(frame, (28, 28), 7, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (42, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if alarm == "FIRE_SMOKE":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 160), -1)
            cv2.putText(
                frame, "EMERGENCY: FIRE / SMOKE", (16, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3,
            )
        elif alarm == "HAND_SIGN":
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 110, 200), -1)
            cv2.putText(
                frame, "ALERT: WORKER ASSISTANCE", (16, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3,
            )
        elif note:
            cv2.putText(frame, note, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    def _draw_overlay(self, frame):
        with self.lock:
            boxes = list(self.boxes)
            points = list(self.hand_points)
            alarm = self.current_alarm
            note = self.preview_note
            camera_open = self.camera_open
        self._paint(frame, boxes, points, alarm, note, camera_open)

    def load_models(self):
        try:
            if YOLO is None or hf_hub_download is None:
                self.model_message = "Install ultralytics and huggingface_hub to enable fire and smoke detection."
            else:
                try:
                    self._models = load_fire_models(hf_hub_download, YOLO, BASE_DIR)
                    self._model = self._models[0]["model"] if self._models else None
                    self.models_ready = bool(self._models)
                    self.model_message = model_message(self._models)
                    print(self.model_message)
                except Exception as exc:
                    self._models = []
                    self._model = None
                    self.model_message = f"Fire model failed to load: {exc}"
                    print(self.model_message)

            if mp is None or not hasattr(mp, "solutions"):
                self.hand_message = "Hand-sign detection needs MediaPipe with Python 3.9–3.11."
                return

            try:
                hands_api = mp.solutions.hands
                self._hands = hands_api.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.4,
                )
                self._hands_still = hands_api.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                )
                self.hand_ready = True
                self.hand_message = (
                    "Worker help: open and close all five fingers 3 times, "
                    "or hold them open for 5 seconds."
                )
                print("Hand-sign model ready.")
            except Exception as exc:
                self.hand_message = f"Hand model failed to load: {exc}"
                print(self.hand_message)
        finally:
            self.models_loaded = True

    def start_camera(self, source):
        with self.lock:
            self.camera_source = parse_source(source)
            self.want_camera = True
            self._reopen = True
            self.camera_message = "Opening camera..."

    def _reset_gesture(self):
        self.finger_count = 0
        self.hand_cycles = 0
        self.hand_hold = 0.0
        self._smooth_points = None
        self._gesture_open = False
        self._open_frames = 0
        self._closed_frames = 0
        self._hold_started = None
        self._hold_grace_until = 0.0
        self._cycle_times = []
        self._last_hand_seen = 0.0
        self._last_points_time = 0.0
        self._count_hist = []
        self._shown_fingers = 0

    def stop_camera(self):
        with self.lock:
            self.want_camera = False
            self._reopen = True
            self.camera_open = False
            self.camera_message = "Live view stopped."
            self.boxes = []
            self.hand_points = []
            self.detections = []
            self.hand_visible = False
            self.preview_note = ""
            self.fire_counter = 0
            self.sign_counter = 0
            self.current_alarm = None
            self._alarm_until = 0.0
            self._pending_boxes = []
            self._last_fire_boxes = []
            self._fire_misses = 0
            self._reset_gesture()
        self._silence()
        self._publish(self._blank("LIVE VIEW STOPPED"))

    def set_speaker(self, enabled, volume):
        with self.lock:
            self.speaker_enabled = bool(enabled)
            self.volume = max(0.0, min(1.0, float(volume)))
            alarm = self.current_alarm
        self._write_tones()
        if not self.speaker_enabled:
            self._silence()
        elif alarm:
            self._play_alarm(alarm)

    def test_speaker(self):
        if self.speaker_error:
            return False, self.speaker_error
        with self.lock:
            if self.current_alarm:
                return False, "An alarm is already using the local speaker."
            if not self.speaker_enabled:
                return False, "Turn on Play alarms on this PC first."
        self._write_tones()
        self._play_file(self._test_path, loop=False)
        with self.lock:
            self.speaker_playing = True
            self._test_until = time.time() + 0.7
        return True, "Test tone is playing on this PC's speakers."

    def _play_file(self, path, loop):
        flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
        if loop:
            flags |= winsound.SND_LOOP
        with self._audio_lock:
            winsound.PlaySound(path, flags)

    def _play_alarm(self, detected_type):
        if not self.speaker_enabled or self.speaker_error:
            with self.lock:
                self.speaker_playing = False
            return
        path = self._fire_path if detected_type == "FIRE_SMOKE" else self._sign_path
        self._play_file(path, loop=True)
        with self.lock:
            self.speaker_playing = True
            self._test_until = 0.0
            self._alarm_until = time.time() + ALARM_HOLD_SECONDS

    def _silence(self):
        try:
            with self._audio_lock:
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        with self.lock:
            self.speaker_playing = False
            self._test_until = 0.0

    def _open_capture(self):
        source = self.camera_source
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def capture_loop(self):
        last_tick = time.time()
        frames = 0
        while not self.stop_event.is_set():
            with self.lock:
                want = self.want_camera
                reopen = self._reopen
                self._reopen = False
            if reopen and self._cap is not None:
                self._cap.release()
                self._cap = None
            if not want:
                time.sleep(0.1)
                continue

            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_capture()
                if not self._cap.isOpened():
                    with self.lock:
                        self.camera_open = False
                        self.camera_message = "No signal. Check the webcam or CCTV address."
                    self._publish(self._blank("NO CAMERA SIGNAL"))
                    time.sleep(1)
                    continue
                with self.lock:
                    self.camera_open = True
                    self.camera_message = "Live view is on."

            ok, frame = self._cap.read()
            if not ok or frame is None:
                with self.lock:
                    self.camera_open = False
                    self.camera_message = "Stream dropped. Reconnecting..."
                self._cap.release()
                self._cap = None
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self._raw = frame
                self._raw_seq += 1
            self._publish(frame)

            frames += 1
            now = time.time()
            if now - last_tick >= 1.0:
                with self.lock:
                    self.fps = frames / (now - last_tick)
                frames = 0
                last_tick = now

    def infer_loop(self):
        seen_seq = -1
        while not self.stop_event.is_set():
            if self._run_pending_job():
                continue
            with self.lock:
                want = self.want_camera
                frame = None if self._raw is None else self._raw.copy()
                seq = self._raw_seq
            if not want or frame is None or seq == seen_seq:
                time.sleep(0.03)
                continue
            seen_seq = seq

            result = self.analyze(frame, still=False)
            raw_points = result["hand_points"]
            points = self._smooth_hand(raw_points)
            if raw_points:
                self._count_hist.append(result["finger_count"])
                self._count_hist = self._count_hist[-3:]
                finger_count = int(round(float(np.median(self._count_hist))))
                self._shown_fingers = finger_count
            elif points:
                finger_count = self._shown_fingers
            else:
                self._count_hist = []
                finger_count = 0
                self._shown_fingers = 0
            sign, note = self._update_gesture(finger_count, bool(points))
            detected = None
            with self.lock:
                self.boxes = result["boxes"]
                self.hand_points = points
                self.preview_note = note
                self.detections = [
                    {"label": box["label"], "conf": round(box["conf"], 3)}
                    for box in result["boxes"]
                ]
                self.hand_visible = bool(points)
                if result["ran_fire"]:
                    if result["fire_found"]:
                        self.fire_counter = min(CONFIRMATION_FRAMES, self.fire_counter + 1)
                        self._fire_misses = 0
                    else:
                        self._fire_misses += 1
                        if self._fire_misses >= 2:
                            self.fire_counter = max(0, self.fire_counter - 1)
                            self._fire_misses = 0
                self.sign_counter = self.hand_cycles
                if self.fire_counter >= CONFIRMATION_FRAMES:
                    detected = "FIRE_SMOKE"
                elif sign:
                    detected = "HAND_SIGN"
                else:
                    detected = None
            self._update_alarm(detected, frame)

    def _run_pending_job(self):
        try:
            frame, holder, done = self._jobs.get_nowait()
        except queue.Empty:
            return False
        try:
            holder["result"] = self._scan_frame(frame)
        except Exception as exc:
            holder["error"] = exc
        finally:
            done.set()
        return True

    def _iou(self, first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if overlap <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return overlap / float(area_a + area_b - overlap)

    def _same_place(self, first, second):
        if self._iou(first, second) >= 0.15:
            return True
        acx, acy = (first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0
        bcx, bcy = (second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0
        return (acx - bcx) ** 2 + (acy - bcy) ** 2 <= 90 ** 2

    def _keep_stable(self, boxes):
        """Show a box after it appears twice in about the same place.

        Fire uses the original place check. Smoke can rise and spread, so a
        smoke box may follow the previous smoke box farther across the frame.
        """
        stable = []
        previous = self._pending_boxes
        prior_xy = []
        prior_smoke = []
        for item in previous:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                prior_xy.append(item[1])
                if item[0] == "smoke":
                    prior_smoke.append(item[1])
            else:
                prior_xy.append(item)
        for box in boxes:
            xyxy = box["xyxy"]
            if box.get("label") == "smoke":
                if box["conf"] >= 0.62 or any(self._smoke_follows(xyxy, old) for old in prior_smoke):
                    stable.append(box)
                continue
            if any(self._same_place(xyxy, old) for old in prior_xy) or box["conf"] >= 0.78:
                stable.append(box)
        self._pending_boxes = [(box["label"], box["xyxy"]) for box in boxes]
        return stable

    def _smoke_follows(self, first, second):
        """True when a new smoke box is the same plume, even if it has drifted."""
        if self._iou(first, second) >= 0.08:
            return True
        acx, acy = (first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0
        bcx, bcy = (second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0
        span = max(
            first[2] - first[0],
            first[3] - first[1],
            second[2] - second[0],
            second[3] - second[1],
            40,
        )
        limit = max(140.0, span * 1.15)
        return (acx - bcx) ** 2 + (acy - bcy) ** 2 <= limit ** 2

    def _accept_smoke(self, conf, pixels, sat, val, skin_ratio, warm_ratio):
        """Keep white, gray, or dark smoke. Reject skin, strong color, and flat surfaces."""
        if pixels < 140 or skin_ratio >= 0.34:
            return False
        white = (sat <= 42) & (val >= 150)
        gray = (sat <= 62) & (val >= 30) & (val <= 220)
        dark = (sat <= 75) & (val >= 18) & (val < 70)
        pale = white | gray
        pale_ratio = float(np.count_nonzero(pale)) / pixels
        dark_ratio = float(np.count_nonzero(dark)) / pixels
        smoke_ratio = float(np.count_nonzero(pale | dark)) / pixels
        spread = float(val.std())
        mean_sat = float(sat.mean())
        if spread < 5.5:
            return False
        if warm_ratio >= 0.40 and smoke_ratio < 0.35:
            return False
        if mean_sat > 80 and smoke_ratio < 0.50:
            return False
        if dark_ratio > pale_ratio and conf < 0.55:
            return False
        if conf >= 0.58:
            return smoke_ratio >= 0.22 and spread >= 6
        if conf >= 0.40:
            return smoke_ratio >= 0.32 and spread >= 8 and skin_ratio < 0.24
        return smoke_ratio >= 0.46 and spread >= 10 and mean_sat <= 50 and skin_ratio < 0.16

    def _accept_hazard(self, frame, label, conf, xyxy):
        """Keep flames with a bright core, and smoke that looks like smoke."""
        x1, y1, x2, y2 = xyxy
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return False
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        pixels = float(hue.size)
        if pixels / float(height * width) > 0.72:
            return False
        warm = ((hue <= 30) | (hue >= 165)) & (sat >= 70) & (val >= 90)
        bright = warm & (val >= 180)
        skin = (hue <= 22) & (sat >= 35) & (sat <= 170) & (val >= 60)
        warm_ratio = float(np.count_nonzero(warm)) / pixels
        bright_ratio = float(np.count_nonzero(bright)) / pixels
        skin_ratio = float(np.count_nonzero(skin)) / pixels
        if label == "fire":
            if pixels < 64:
                return False
            if bright_ratio >= 0.03:
                return True
            return warm_ratio >= 0.12 and bright_ratio >= 0.012 and conf >= 0.45
        return self._accept_smoke(conf, pixels, sat, val, skin_ratio, warm_ratio)

    def analyze(self, frame, still=False):
        fire_found = False
        boxes = []
        hand_points = []
        finger_count = 0
        now = time.time()
        run_fire = still or (now - self._last_yolo) >= 0.12
        with self._infer_lock:
            if self._models and run_fire:
                imgsz = 416 if len(self._models) > 1 else 512
                found = []
                for source in self._models:
                    model = source["model"]
                    results = model(frame, imgsz=imgsz, conf=0.22, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            label = canonical_label(model.names[int(box.cls)])
                            conf = float(box.conf)
                            if label == "fire" and conf < FIRE_CONFIDENCE:
                                continue
                            if label == "smoke" and conf < SMOKE_CONFIDENCE:
                                continue
                            if label not in ("fire", "smoke"):
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            xyxy = (x1, y1, x2, y2)
                            if not self._accept_hazard(frame, label, conf, xyxy):
                                continue
                            found.append({
                                "xyxy": xyxy,
                                "label": label,
                                "conf": conf,
                                "source": source["name"],
                            })
                found = merge_boxes(found)
                if still:
                    boxes = found
                else:
                    boxes = self._keep_stable(found)
                    self._last_fire_boxes = boxes
                    self._last_yolo = time.time()
                fire_found = bool(boxes)
            elif not still and (now - self._last_yolo) <= 0.45:
                boxes = list(self._last_fire_boxes)

            detector = self._hands_still if still and self._hands_still is not None else self._hands
            if detector is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                hand_results = detector.process(rgb)
                if hand_results.multi_hand_landmarks:
                    best_count = -1
                    for hand_lms in hand_results.multi_hand_landmarks:
                        count = self._finger_count(hand_lms)
                        pts = [
                            (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                            for lm in hand_lms.landmark
                        ]
                        if count > best_count:
                            best_count = count
                            hand_points = pts
                            finger_count = count
        open_palm = finger_count >= 5
        return {
            "fire_found": fire_found,
            "ran_fire": run_fire,
            "sign_found": bool(still and open_palm),
            "boxes": boxes,
            "hand_points": hand_points,
            "finger_count": finger_count,
            "note": "OPEN PALM" if still and open_palm else "",
        }

    def _smooth_hand(self, points):
        now = time.time()
        if len(points) < 21:
            if self._smooth_points and now - self._last_points_time < 0.28:
                return list(self._smooth_points)
            self._smooth_points = None
            return []
        self._last_points_time = now
        previous = self._smooth_points
        if not previous or len(previous) != len(points):
            self._smooth_points = points
            return points
        blended = []
        for (x, y), (px, py) in zip(points, previous):
            if abs(x - px) + abs(y - py) > 140:
                blended.append((x, y))
            else:
                blended.append((int(0.78 * x + 0.22 * px), int(0.78 * y + 0.22 * py)))
        self._smooth_points = blended
        return blended

    def _finger_count(self, hand_landmarks):
        """Count straight fingers. Joint angles stay valid when the hand rotates."""
        landmarks = hand_landmarks.landmark

        def distance(first, second):
            start = landmarks[first]
            end = landmarks[second]
            return (
                (start.x - end.x) ** 2
                + (start.y - end.y) ** 2
                + (start.z - end.z) ** 2
            ) ** 0.5

        def bend(mcp, joint, tip):
            joint_pt = landmarks[joint]
            toward_mcp = (
                landmarks[mcp].x - joint_pt.x,
                landmarks[mcp].y - joint_pt.y,
                landmarks[mcp].z - joint_pt.z,
            )
            toward_tip = (
                landmarks[tip].x - joint_pt.x,
                landmarks[tip].y - joint_pt.y,
                landmarks[tip].z - joint_pt.z,
            )
            left = math.sqrt(sum(part * part for part in toward_mcp))
            right = math.sqrt(sum(part * part for part in toward_tip))
            if left < 1e-6 or right < 1e-6:
                return 0.0
            cosine = sum(a * b for a, b in zip(toward_mcp, toward_tip)) / (left * right)
            cosine = max(-1.0, min(1.0, cosine))
            return math.degrees(math.acos(cosine))

        count = 0
        thumb_straight = bend(2, 3, 4) > 145
        thumb_out = distance(4, 17) > distance(2, 17) * 1.04
        if thumb_straight and thumb_out:
            count += 1
        for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            straight = bend(mcp, pip, tip) > 150
            extended = distance(tip, 0) > distance(pip, 0) * 1.06
            if straight and extended:
                count += 1
        return count

    def _update_gesture(self, finger_count, hand_seen):
        now = time.time()
        sign = False
        cycled = False
        held_full = False
        with self.lock:
            if not hand_seen:
                self.finger_count = 0
                self._open_frames = 0
                self._closed_frames = 0
                if self._last_hand_seen and now - self._last_hand_seen > 0.6:
                    self._gesture_open = False
                if self._hold_started is not None and now > self._hold_grace_until:
                    self._hold_started = None
            else:
                self._last_hand_seen = now
                self.finger_count = finger_count
                if finger_count >= 4:
                    self._open_frames += 1
                    self._closed_frames = 0
                    if self._hold_started is None:
                        self._hold_started = now
                    self._hold_grace_until = now + 0.5
                elif finger_count == 3 and self._hold_started is not None and now <= self._hold_grace_until:
                    self._open_frames = 0
                    self._closed_frames = 0
                else:
                    self._open_frames = 0
                    if finger_count <= 2:
                        self._closed_frames += 1
                        if self._closed_frames >= 2:
                            self._hold_started = None
                    else:
                        self._closed_frames = 0

                if self._open_frames >= 2:
                    self._gesture_open = True
                if self._gesture_open and self._closed_frames >= 2:
                    self._gesture_open = False
                    self._open_frames = 0
                    self._closed_frames = 0
                    self._cycle_times.append(now)

                cutoff = now - CYCLE_WINDOW_SECONDS
                self._cycle_times = [stamp for stamp in self._cycle_times if stamp >= cutoff]
                cycled = len(self._cycle_times) >= OPEN_CLOSE_CYCLES
                if cycled:
                    sign = True
                    self._cycle_times = []
                    self._gesture_open = False

                held_full = (
                    self._hold_started is not None
                    and (now - self._hold_started) >= HOLD_FINGERS_SECONDS
                )
                if held_full:
                    sign = True

            hold = 0.0
            if self._hold_started is not None:
                hold = min(HOLD_FINGERS_SECONDS, now - self._hold_started)
            cycles = min(OPEN_CLOSE_CYCLES, len(self._cycle_times))
            self.hand_hold = hold
            self.hand_cycles = cycles

            if held_full:
                note = "HELP: FIVE FINGERS HELD"
            elif cycled:
                note = "HELP: OPEN-CLOSE x3"
            elif hand_seen and hold > 0.15 and finger_count >= 3:
                note = f"HOLD {hold:.1f}s / 5s"
            elif hand_seen and cycles:
                note = f"OPEN-CLOSE {cycles}/3"
            elif hand_seen:
                note = f"FINGERS {finger_count}/5"
            else:
                note = ""
        return sign, note

    def _fit(self, frame, max_side=960):
        height, width = frame.shape[:2]
        scale = max_side / float(max(height, width))
        if scale >= 1:
            return frame
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(frame, size)

    def detect_image(self, frame):
        fitted = self._fit(frame, 960)
        done = threading.Event()
        holder = {}
        self._jobs.put((fitted, holder, done))
        if not done.wait(45):
            raise TimeoutError("Detection took too long. Try a smaller image.")
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    def _scan_frame(self, frame):
        result = self.analyze(frame, still=True)
        alarm = None
        if result["fire_found"]:
            alarm = "FIRE_SMOKE"
        elif result["sign_found"]:
            alarm = "HAND_SIGN"
        view = frame.copy()
        self._paint(view, result["boxes"], result["hand_points"], alarm, result["note"], live=False)
        ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            raise RuntimeError("Could not encode the detected image.")
        saved = None
        if alarm:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved = f"{alarm}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(LOG_DIR, saved), view)
            self._cue_scan(alarm)
        return {
            "ok": True,
            "alarm": alarm,
            "hand_sign": result["sign_found"],
            "hand_visible": bool(result["hand_points"]),
            "note": result["note"],
            "detections": [
                {
                    "label": box["label"],
                    "conf": round(box["conf"], 3),
                    "box": [int(v) for v in box["xyxy"]],
                }
                for box in result["boxes"]
            ],
            "models_ready": self.models_ready,
            "hand_ready": self.hand_ready,
            "model_message": self.model_message,
            "hand_message": self.hand_message,
            "width": int(view.shape[1]),
            "height": int(view.shape[0]),
            "saved": saved,
            "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
        }

    def _cue_scan(self, detected_type):
        if self.speaker_error or not self.speaker_enabled:
            return
        with self.lock:
            if self.current_alarm:
                return
        path = self._fire_path if detected_type == "FIRE_SMOKE" else self._sign_path
        self._play_file(path, loop=False)
        with self.lock:
            self.speaker_playing = True
            self._test_until = time.time() + 1.6

    def _update_alarm(self, detected_type, frame):
        now = time.time()
        with self.lock:
            current = self.current_alarm
            alarm_until = self._alarm_until

        if detected_type:
            fresh = current != detected_type
            if fresh or now >= alarm_until:
                self._play_alarm(detected_type)
            if fresh:
                with self.lock:
                    boxes = list(self.boxes)
                    points = list(self.hand_points)
                    note = self.preview_note
                shot = frame.copy()
                self._paint(shot, boxes, points, detected_type, note, live=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = os.path.join(LOG_DIR, f"{detected_type}_{timestamp}.jpg")
                cv2.imwrite(log_path, shot)
                print(f"[ALERT TRIGGERED] {detected_type} -> Snapshot saved to {log_path}")
            with self.lock:
                self.last_alert_time = now
                self.current_alarm = detected_type
        elif current and now >= alarm_until:
            self._silence()
            print(f"[CLEAR] {current} resolved after {ALARM_HOLD_SECONDS}s. Speaker muted.")
            with self.lock:
                self.current_alarm = None
                self._alarm_until = 0.0

    def snapshot(self):
        with self.lock:
            playing = False
            if self.current_alarm and self.speaker_enabled and time.time() < self._alarm_until:
                playing = True
            elif time.time() < self._test_until:
                playing = True
            else:
                self.speaker_playing = False
            return {
                "camera_open": self.camera_open,
                "streaming": self.want_camera,
                "source": str(self.camera_source),
                "camera_message": self.camera_message,
                "models_ready": self.models_ready,
                "model_message": self.model_message,
                "hand_ready": self.hand_ready,
                "hand_message": self.hand_message,
                "alarm": self.current_alarm,
                "speaker_enabled": self.speaker_enabled,
                "speaker_playing": playing,
                "speaker_error": self.speaker_error,
                "volume": self.volume,
                "fire_counter": self.fire_counter,
                "sign_counter": self.sign_counter,
                "finger_count": self.finger_count,
                "hand_cycles": self.hand_cycles,
                "hand_hold": round(self.hand_hold, 1),
                "fps": round(self.fps, 1),
                "models_loaded": self.models_loaded,
                "detections": list(self.detections),
                "hand_visible": self.hand_visible,
                "preview_note": self.preview_note,
            }


monitor = SafetyMonitor()


def incident_kind(name):
    upper = name.upper()
    if upper.startswith("FIRE_SMOKE"):
        return "FIRE_SMOKE"
    if upper.startswith("HAND_SIGN"):
        return "HAND_SIGN"
    return "OTHER"


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def incident_path(name):
    safe = os.path.basename(str(name or ""))
    if not safe or safe != str(name) or not safe.lower().endswith(IMAGE_EXTENSIONS):
        return None
    root = os.path.realpath(LOG_DIR)
    path = os.path.realpath(os.path.join(root, safe))
    if os.path.dirname(path) != root:
        return None
    return path


def list_incidents():
    rows = []
    for name in os.listdir(LOG_DIR):
        path = incident_path(name)
        if path is None or not os.path.isfile(path):
            continue
        rows.append({
            "name": name,
            "mtime": os.path.getmtime(path),
            "kind": incident_kind(name),
        })
    rows.sort(key=lambda item: item["mtime"], reverse=True)
    return rows[:24]


def delete_incident(name):
    path = incident_path(name)
    if path is None or not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def delete_all_incidents():
    removed = 0
    for name in os.listdir(LOG_DIR):
        if delete_incident(name):
            removed += 1
    return removed


def delete_key_ok():
    data = request.get_json(silent=True) or {}
    given = str(data.get("key", ""))
    if len(given) != len(DELETE_KEY):
        return False
    return hmac.compare_digest(given, DELETE_KEY)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with monitor.lock:
                frame = monitor.jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/api/status")
def api_status():
    return jsonify(monitor.snapshot())


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    source = data.get("source", monitor.camera_source)
    if str(source).strip() == "":
        return jsonify({"ok": False, "message": "Enter a camera number or CCTV URL."}), 400
    monitor.start_camera(source)
    return jsonify({"ok": True, "message": "Opening live view."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    monitor.stop_camera()
    return jsonify({"ok": True, "message": "Live view stopped. Speaker is quiet."})


@app.route("/api/speaker", methods=["POST"])
def api_speaker():
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", monitor.speaker_enabled)
    volume = data.get("volume", monitor.volume)
    try:
        monitor.set_speaker(enabled, volume)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "message": "Volume must be a number from 0 to 1."}), 400
    return jsonify({"ok": True})


@app.route("/api/test-sound", methods=["POST"])
def api_test_sound():
    ok, message = monitor.test_speaker()
    return jsonify({"ok": ok, "message": message}), (200 if ok else 400)


@app.route("/api/incidents")
def api_incidents():
    return jsonify(list_incidents())


@app.route("/api/incidents/<path:name>", methods=["DELETE"])
def api_delete_incident(name):
    if not delete_key_ok():
        return jsonify({"ok": False, "message": "Wrong key. The image was not deleted."}), 403
    if not delete_incident(name):
        return jsonify({"ok": False, "message": "That snapshot is already gone."}), 404
    return jsonify({"ok": True, "message": "Snapshot deleted."})


@app.route("/api/incidents", methods=["DELETE"])
def api_delete_incidents():
    if not delete_key_ok():
        return jsonify({"ok": False, "message": "Wrong key. Nothing was deleted."}), 403
    removed = delete_all_incidents()
    if removed == 0:
        return jsonify({"ok": True, "message": "No snapshots to delete."})
    label = "snapshot" if removed == 1 else "snapshots"
    return jsonify({"ok": True, "message": f"Deleted {removed} {label}."})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if not monitor.models_loaded:
        return jsonify({
            "ok": False,
            "message": "Detection models are still loading. Wait a few seconds and try again.",
        }), 503
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "message": "Choose an image to scan."}), 400
    raw = upload.read()
    if not raw:
        return jsonify({"ok": False, "message": "That image file is empty."}), 400
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({
            "ok": False,
            "message": "That file is not a readable image. Use JPG, PNG, or WEBP.",
        }), 400
    try:
        return jsonify(monitor.detect_image(frame))
    except Exception as exc:
        return jsonify({"ok": False, "message": f"Detection failed: {exc}"}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"ok": False, "message": "Image must be 12 MB or smaller."}), 413


@app.route("/incidents/<path:name>")
def incident_file(name):
    safe = os.path.basename(name)
    return send_from_directory(LOG_DIR, safe)


def main():
    parser = argparse.ArgumentParser(description="AI Safety Monitor live view")
    parser.add_argument("--camera", default="0", help="0 for webcam, or an http/rtsp CCTV URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    threading.Thread(target=monitor.load_models, daemon=True).start()
    threading.Thread(target=monitor.capture_loop, daemon=True).start()
    threading.Thread(target=monitor.infer_loop, daemon=True).start()
    monitor.start_camera(args.camera)

    url = f"http://{args.host}:{args.port}"
    print(f"Live view: {url}")
    print("Alarms play on this PC's speakers for 30 seconds. Close this window or press Ctrl+C to quit.")
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
