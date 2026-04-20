import cv2
import numpy as np
from datetime import datetime
import face_recognition
import pickle
import os
import threading

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_FILE    = 'face_database.pkl'
ALARM_COOLDOWN   = 10      # seconds between unknown-face alerts
FLASH_DURATION   = 8       # frames to show red flash overlay
LOG_MAX_ENTRIES  = 6       # lines shown in on-screen log
SNAPSHOT_DIR     = 'snapshots'

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
known_faces     = {}
known_names     = []
known_encodings = []

if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'rb') as f:
        known_faces = pickle.load(f)
    for name, encodings_list in known_faces.items():
        for encoding in encodings_list:
            known_encodings.append(encoding)
            known_names.append(name)
    print(f"✓ Loaded {len(known_faces)} authorized face(s)")
    print(f"  Total reference photos: {len(known_encodings)}")
else:
    print("⚠ No face database found. Run enroll_faces.py first!")

# ── Shared state ──────────────────────────────────────────────────────────────
lock   = threading.Lock()
shared = {
    "frame":   None,
    "results": [],
    "running": True,
}

# ── Recognition worker ────────────────────────────────────────────────────────
def recognition_worker():
    last_frame_id = None

    while True:
        with lock:
            if not shared["running"]:
                break
            frame     = shared["frame"]
            frame_id  = id(frame)

        if frame is None or frame_id == last_frame_id:
            threading.Event().wait(0.01)
            continue

        last_frame_id = frame_id

        if len(known_encodings) == 0:
            continue

        small     = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small, model="hog")
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        results = []
        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4

            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            name    = "UNKNOWN"
            color   = (0, 0, 255)

            if True in matches:
                name  = known_names[matches.index(True)]
                color = (0, 255, 0)

            results.append((name, color, left, top, right, bottom))

        with lock:
            shared["results"] = results

worker = threading.Thread(target=recognition_worker, daemon=True)
worker.start()

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam")
    with lock:
        shared["running"] = False
    exit()

print("Security monitor running.")
print("Press 's' to save snapshot manually | 'q' / Esc to quit")

# ── State ─────────────────────────────────────────────────────────────────────
snapshot_count   = 0
flash_frames     = 0           # counts down when alarm active
last_alarm_time  = {}          # name -> datetime, cooldown tracker
detection_log    = []          # list of strings for on-screen log
seen_this_cycle  = set()       # names seen in current results, to avoid log spam

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def log_event(msg):
    detection_log.append(f"{timestamp()}  {msg}")
    if len(detection_log) > LOG_MAX_ENTRIES:
        detection_log.pop(0)
    print(msg)

def save_snapshot(frame, label):
    global snapshot_count
    snapshot_count += 1
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(SNAPSHOT_DIR, f"{label}_{ts}.jpg")
    cv2.imwrite(fname, frame)
    print(f"✓ Snapshot saved: {fname}")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    with lock:
        shared["frame"] = frame.copy()
        results         = list(shared["results"])

    now          = datetime.now()
    display      = frame.copy()
    unknown_seen = False
    current_seen = set()

    # Draw face boxes
    for name, color, left, top, right, bottom in results:
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        cv2.rectangle(display, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
        cv2.putText(display, name, (left + 6, bottom + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        current_seen.add(name)

        if name == "UNKNOWN":
            unknown_seen = True

    # Alarm + auto-snapshot for new detections
    for name in current_seen - seen_this_cycle:
        last_t = last_alarm_time.get(name)
        if last_t is None or (now - last_t).total_seconds() > ALARM_COOLDOWN:
            last_alarm_time[name] = now
            if name == "UNKNOWN":
                log_event("!! UNKNOWN face detected")
                save_snapshot(frame, "UNKNOWN")
                flash_frames = FLASH_DURATION
            else:
                log_event(f">> {name}")

    seen_this_cycle = current_seen

    # Red flash overlay when unknown detected
    if flash_frames > 0:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                      (0, 0, 200), -1)
        alpha   = 0.25 * (flash_frames / FLASH_DURATION)
        display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)
        flash_frames -= 1

    # ── HUD ───────────────────────────────────────────────────────────────────
    h, w = display.shape[:2]

    # Top bar: timestamp + authorized count
    cv2.rectangle(display, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(display, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    auth_text = f"Authorized: {len(known_faces)}"
    cv2.putText(display, auth_text,
                (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

    # Status indicator (top right corner dot)
    status_color = (0, 0, 255) if unknown_seen else (0, 255, 0)
    cv2.circle(display, (w - 20, 18), 7, status_color, -1)

    # Detection log (bottom of screen, dynamic height based on entries)
    if detection_log:
        log_bg_h = len(detection_log) * 22 + 10
        cv2.rectangle(display, (0, h - log_bg_h), (w, h), (0, 0, 0), -1)
        for i, entry in enumerate(detection_log):
            color = (0, 100, 255) if "UNKNOWN" in entry else (180, 255, 180)
            cv2.putText(display, entry,
                        (10, h - log_bg_h + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    cv2.imshow('Security Monitor', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_snapshot(display, "manual")
    elif key == ord('q') or key == ord('Q') or key == 27:
        break

    # Also break if the window was closed with the X button
    if cv2.getWindowProperty('Security Monitor', cv2.WND_PROP_VISIBLE) < 1:
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
with lock:
    shared["running"] = False

cap.release()
cv2.destroyAllWindows()
print(f"\nSnapshots saved to: {SNAPSHOT_DIR}/")
print("Monitor closed")