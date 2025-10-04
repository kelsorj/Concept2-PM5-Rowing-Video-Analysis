#!/usr/bin/env python3
# Py3Row USB + FaceTime video capture with shared timestamps

import sys
import os
import csv
import json
import time
import datetime
import threading
import queue

import cv2

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))
from pyrow import pyrow
from pyrow.ergmanager import ErgManager

# macOS USB permission: require sudo
if os.geteuid() != 0:
    print("ERROR: This script requires sudo on macOS for USB HID access.")
    print("Run: sudo ./rowing_env/bin/python3 py3row_usb_video_capture.py -- ...")
    exit(1)

def now_ns():
    return time.perf_counter_ns()

def ns_to_iso(ts_ns_offset, wall_anchor):
    # Convert monotonic offset (ns) to wall-clock ISO using session start anchor
    return (wall_anchor + datetime.timedelta(microseconds=ts_ns_offset/1000)).isoformat(timespec="milliseconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record PM5 data (Py3Row) and video with synchronized timestamps")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (FaceTime HD is often 0)")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS")
    parser.add_argument("--out-prefix", default=datetime.datetime.now().strftime("py3rowcap_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # Shared time anchors
    start_wall = datetime.datetime.now()
    start_ns = now_ns()

    # Outputs
    video_path = f"{args.out_prefix}.mp4"
    frames_csv_path = f"{args.out_prefix}_frames.csv"
    pm5_csv_path = f"{args.out_prefix}_pm5.csv"

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # CSV writers
    f_frames = open(frames_csv_path, 'w', newline='')
    frame_writer = csv.DictWriter(f_frames, fieldnames=["frame_idx","ts_ns","ts_iso"])
    frame_writer.writeheader()

    f_pm5 = open(pm5_csv_path, 'w', newline='')
    pm_fields = [
        'ts_ns','ts_iso',
        'elapsed_s','distance_m','spm','power','pace','calhr','calories',
        'strokestate','status','forceplot_json'
    ]
    pm_writer = csv.DictWriter(f_pm5, fieldnames=pm_fields)
    pm_writer.writeheader()

    # Thread-safe queue for PM5 updates (avoid writing CSV in callback thread)
    pm_q = queue.Queue(maxsize=2048)

    # Stroke aggregation state (kept in callback closure)
    stroke_state = {
        'accum': [],
        'last_state': None,
        'stroke_idx': 0,
    }

    def new_erg_callback(erg):
        ts_ns_off = now_ns() - start_ns
        print(f"✅ PM5 connected at {ns_to_iso(ts_ns_off, start_wall)}")

    def update_erg_callback(erg):
        # Called ~5 Hz by ErgManager
        ts_ns_off = now_ns() - start_ns
        ts_iso = ns_to_iso(ts_ns_off, start_wall)
        m = getattr(erg, 'data', {}) or {}

        # Accumulate forceplot across a stroke
        fp = m.get('forceplot', []) or []
        if fp:
            stroke_state['accum'].extend(fp)
        curr = m.get('strokestate')
        prev = stroke_state['last_state']

        completed_forceplot = None
        if prev == 'Drive' and curr == 'Recovery' and stroke_state['accum']:
            stroke_state['stroke_idx'] += 1
            completed_forceplot = stroke_state['accum'][:]
            stroke_state['accum'].clear()

        stroke_state['last_state'] = curr

        pm_q.put({
            'ts_ns': ts_ns_off,
            'ts_iso': ts_iso,
            'm': m,
            'forceplot_json': json.dumps(completed_forceplot) if completed_forceplot else ""
        })

    print("🔍 Scanning for PM5 devices...")
    ergman = ErgManager(pyrow,
                        add_callback=new_erg_callback,
                        update_callback=update_erg_callback,
                        check_rate=1,
                        update_rate=0.2)
    print("✅ Py3Row ErgManager running. Recording… Press Ctrl+C to stop.")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame timestamp from shared clock
            ts_ns_off = now_ns() - start_ns
            ts_iso = ns_to_iso(ts_ns_off, start_wall)

            # Burn-in timestamp (bottom-right)
            cv2.putText(frame, ts_iso, (width - 320, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out.write(frame)
            frame_writer.writerow({"frame_idx": frame_idx, "ts_ns": ts_ns_off, "ts_iso": ts_iso}); f_frames.flush()
            frame_idx += 1

            # Drain PM5 queue
            while not pm_q.empty():
                item = pm_q.get_nowait()
                m = item['m']
                pm_writer.writerow({
                    'ts_ns': int(item['ts_ns']),
                    'ts_iso': item['ts_iso'],
                    'elapsed_s': m.get('time', 0),
                    'distance_m': m.get('distance', 0),
                    'spm': m.get('spm', 0),
                    'power': m.get('power', 0),
                    'pace': m.get('pace', 0),
                    'calhr': m.get('calhr', 0),
                    'calories': m.get('calories', 0),
                    'strokestate': m.get('strokestate', ''),
                    'status': m.get('status', ''),
                    'forceplot_json': item['forceplot_json'],
                }); f_pm5.flush()

            time.sleep(max(0.0, (1.0 / fps) - 0.001))
    except KeyboardInterrupt:
        print("\n⏹️  Stopping…")
    finally:
        ergman.stop()
        cap.release(); out.release()
        f_frames.close(); f_pm5.close()
        print("Saved:")
        print("  Video:", video_path)
        print("  Frames:", frames_csv_path)
        print("  PM5:", pm5_csv_path)

if __name__ == "__main__":
    main()


