
"""Smart Cam: live YOLO detection with tracking IDs, a restricted zone, alerts and a log.

Run:      python smart_cam.py
Options:  python smart_cam.py --source 1            (another camera)
          python smart_cam.py --source video.mp4    (a video file)
          python smart_cam.py --model yolo26s.pt    (more accurate, slower)
          python smart_cam.py --classes 0 2 16      (person, car, dog)

Keys:     drag mouse = draw the restricted zone
          c = clear zone, r = reset counts, s = save snapshot, q = quit
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0")
parser.add_argument("--model", default="yolo26n.pt")
parser.add_argument("--classes", nargs="+", type=int, default=[0])  # 0 = person
parser.add_argument("--conf", type=float, default=0.4)
args = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source

zone = None  # (x1, y1, x2, y2)
drag_start = None


def on_mouse(event, x, y, flags, param):
    global zone, drag_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drag_start:
        x0, y0 = drag_start
        zone = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None


def label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


model = YOLO(args.model)
names = model.names
out = Path("alerts")
out.mkdir(exist_ok=True)
log_file = out / "log.csv"
if not log_file.exists():
    log_file.write_text("time,track_id,class\n")

seen = set()  # every track ID seen so far
last_alert = {}  # track ID -> time of last alert
fps, prev = 0.0, time.time()

cap = cv2.VideoCapture(source)
win = "Smart Cam"
cv2.namedWindow(win)
cv2.setMouseCallback(win, on_mouse)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    r = model.track(frame, persist=True, conf=args.conf, classes=args.classes, verbose=False)[0]
    in_view, new_alerts, breach = 0, [], False

    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid, cls in zip(
            r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()
        ):
            in_view += 1
            seen.add(tid)
            fx, fy = (x1 + x2) // 2, y2  # bottom centre of the box (where the feet are)
            inside = bool(zone) and zone[0] <= fx <= zone[2] and zone[1] <= fy <= zone[3]
            color = (0, 0, 255) if inside else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label(frame, f"{names[cls]} #{tid}", (x1, max(y1 - 8, 15)), color)
            if inside:
                breach = True
                if time.time() - last_alert.get(tid, 0) > 10:  # 10 second cooldown per ID
                    last_alert[tid] = time.time()
                    new_alerts.append((tid, names[cls]))

    if zone:
        cv2.rectangle(frame, zone[:2], zone[2:], (0, 0, 255) if breach else (255, 200, 0), 2)
    else:
        label(frame, "Drag the mouse to draw a restricted zone", (10, frame.shape[0] - 15), (255, 255, 255))

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
    prev = now
    label(frame, f"FPS {fps:.0f}   In view {in_view}   Total seen {len(seen)}", (10, 25), (255, 255, 255))

    if breach:
        cv2.rectangle(frame, (0, 40), (frame.shape[1], 80), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: ZONE BREACH", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for tid, cname in new_alerts:
        stamp = datetime.now()
        cv2.imwrite(str(out / f"{stamp:%Y%m%d_%H%M%S}_id{tid}.jpg"), frame)
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([stamp.isoformat(timespec="seconds"), tid, cname])
        if winsound:
            winsound.Beep(1000, 150)

    cv2.imshow(win, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        zone = None
    elif key == ord("r"):
        seen.clear()
        last_alert.clear()
    elif key == ord("s"):
        cv2.imwrite(str(out / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"), frame)

cap.release()
cv2.destroyAllWindows()
