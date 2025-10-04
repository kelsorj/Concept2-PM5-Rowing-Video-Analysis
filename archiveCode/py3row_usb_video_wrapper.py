#!/usr/bin/env python3
# Wrapper: run the known-good py3row_usb_capture.py in parallel with FaceTime video capture.
# Uses the same wall-clock timestamp (datetime.now) overlay on frames for easy sync.

import os, sys, time, csv, datetime, subprocess, pathlib, argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description="Run py3row_usb_capture + video with wall-clock timestamps")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-prefix", default=datetime.datetime.now().strftime("py3rowwrap_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    base = os.path.basename(args.out_prefix)
    session_dir = args.out_prefix
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, f"{base}.mp4")
    frames_csv = os.path.join(session_dir, f"{base}_frames.csv")

    # Start data capture subprocess (must be run with sudo outside)
    proc = subprocess.Popen([sys.executable, "py3row_usb_capture.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        proc.terminate()
        raise SystemExit(f"Could not open camera {args.camera}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    fw = open(frames_csv, 'w', newline=''); writer = csv.DictWriter(fw, fieldnames=["frame_idx","ts_iso"]); writer.writeheader()

    print("Recording… Press Ctrl+C to stop")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            ts_iso = datetime.datetime.now().isoformat(timespec='milliseconds')
            cv2.putText(frame, ts_iso, (w-320, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out.write(frame)
            writer.writerow({"frame_idx": frame_idx, "ts_iso": ts_iso}); fw.flush()
            frame_idx += 1
            time.sleep(max(0.0, (1.0/fps) - 0.001))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); out.release(); fw.close()
        try:
            proc.terminate()
        except Exception:
            pass

    # After stop, copy the latest py3row outputs into the session folder
    root = pathlib.Path('.')
    parsed = sorted(root.glob('pm5_py3row_parsed_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    raw = sorted(root.glob('pm5_py3row_raw_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if parsed:
        parsed[0].rename(pathlib.Path(session_dir) / parsed[0].name)
        print("Saved:", parsed[0].name, "→", session_dir)
    if raw:
        raw[0].rename(pathlib.Path(session_dir) / raw[0].name)
        print("Saved:", raw[0].name, "→", session_dir)
    print("Video:", video_path)

if __name__ == '__main__':
    main()


