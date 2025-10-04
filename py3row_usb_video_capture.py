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
import subprocess
import tempfile

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

    # Outputs in a dedicated folder named after the prefix
    base = os.path.basename(args.out_prefix)
    session_dir = args.out_prefix
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, f"{base}.mp4")
    frames_csv_path = os.path.join(session_dir, f"{base}_frames.csv")
    pm5_csv_path = os.path.join(session_dir, f"{base}_pm5.csv")
    raw_csv_path = os.path.join(session_dir, f"{base}_raw.csv")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(args.fps)
    
    # Create temporary directory for frame files
    temp_dir = tempfile.mkdtemp(prefix="rowing_frames_")
    print(f"📁 Using temporary directory: {temp_dir}")
    
    # FFmpeg command for video encoding with metadata
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Enable fast start for web playback
        '-metadata', f'creation_time={start_wall.isoformat()}',
        '-metadata', f'session_start={start_wall.isoformat()}',
        video_path
    ]
    
    # Start FFmpeg process
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"🎥 FFmpeg process started for video encoding")

    # CSV writers
    f_frames = open(frames_csv_path, 'w', newline='')
    frame_writer = csv.DictWriter(f_frames, fieldnames=["frame_idx","ts_ns","ts_iso","frame_file"])
    frame_writer.writeheader()

    f_pm5 = open(pm5_csv_path, 'w', newline='')
    pm_fields = [
        'ts_ns','ts_iso',
        'elapsed_s','distance_m','spm','power','pace','calhr','calories',
        'strokestate','status','forceplot_current_json','forceplot_complete_json'
    ]
    pm_writer = csv.DictWriter(f_pm5, fieldnames=pm_fields)
    pm_writer.writeheader()

    # Raw monitor JSON stream (one row per update)
    f_raw = open(raw_csv_path, 'w', newline='')
    raw_writer = csv.DictWriter(f_raw, fieldnames=['ts_ns','ts_iso','raw_json'])
    raw_writer.writeheader()

    # Thread-safe queue for PM5 updates (avoid writing CSV in callback thread)
    pm_q = queue.Queue(maxsize=2048)

    # Stroke aggregation state (kept in callback closure)
    stroke_state = {
        'accum': [],
        'last_state': None,
        'stroke_idx': 0,
        'last_nonempty_fp': [],
    }
    fp_lock = threading.Lock()
    stop_evt = threading.Event()
    erg_ref = {'erg': None}

    def new_erg_callback(erg):
        ts_ns_off = now_ns() - start_ns
        print(f"✅ PM5 connected at {ns_to_iso(ts_ns_off, start_wall)}")
        erg_ref['erg'] = erg

    def update_erg_callback(erg):
        # Called ~5 Hz by ErgManager
        ts_ns_off = now_ns() - start_ns
        ts_iso = ns_to_iso(ts_ns_off, start_wall)
        m = getattr(erg, 'data', {}) or {}

        # Get forceplot data from monitor (ErgManager already calls get_monitor(forceplot=True))
        # Match the working script approach exactly
        fp = m.get('forceplot', []) or []
        curr_stroke_state = m.get('strokestate', '')
        power = m.get('power', 0)
        spm = m.get('spm', 0)
        
        # Debug: Print when we get forceplot data
        if fp and len(fp) > 0:
            print(f"🎯 Forceplot data received: {len(fp)} samples, state={curr_stroke_state}, power={power}W, spm={spm}")
        else:
            # Print occasionally when no forceplot data (every 10th call to avoid spam)
            if hasattr(update_erg_callback, 'call_count'):
                update_erg_callback.call_count += 1
            else:
                update_erg_callback.call_count = 1
            
            if update_erg_callback.call_count % 10 == 0:
                print(f"⏳ PM5 update #{update_erg_callback.call_count}: state={curr_stroke_state}, power={power}W, spm={spm}, forceplot_len={len(fp)}")

        # Accumulate forceplot across a stroke
        if fp and len(fp) > 0:
            with fp_lock:
                stroke_state['accum'].extend(fp)
                stroke_state['last_nonempty_fp'] = fp

        prev_stroke_state = stroke_state['last_state']

        # Detect stroke completion (transition from Drive to Recovery) - match working script
        completed_forceplot = None
        if (prev_stroke_state == 'Drive' and 
            curr_stroke_state == 'Recovery' and 
            stroke_state['accum']):
            with fp_lock:
                stroke_state['stroke_idx'] += 1
                completed_forceplot = stroke_state['accum'][:]
                print(f"🎯 Complete stroke #{stroke_state['stroke_idx']} captured: {len(completed_forceplot)} force points")
                stroke_state['accum'].clear()
                stroke_state['last_nonempty_fp'] = []

        stroke_state['last_state'] = curr_stroke_state

        pm_q.put({
            'ts_ns': ts_ns_off,
            'ts_iso': ts_iso,
            'm': m,
            'forceplot_current_json': json.dumps(stroke_state.get('last_nonempty_fp') or fp) if (stroke_state.get('last_nonempty_fp') or fp) else "",
            'forceplot_complete_json': json.dumps(completed_forceplot) if completed_forceplot else "",
            'raw_json': json.dumps(m)
        })

    print("🔍 Scanning for PM5 devices...")
    print("💡 TIP: Start a 'Just Row' workout on your PM5 to get forceplot data!")
    print("💡 Forceplot data is only available during active rowing strokes.")
    print("💡 If no forceplot data appears, try different workout modes or check PM5 firmware.")
    ergman = ErgManager(pyrow,
                        add_callback=new_erg_callback,
                        update_callback=update_erg_callback,
                        check_rate=1,
                        update_rate=0.2)  # Match working script: 5Hz updates
    print("✅ Py3Row ErgManager running. Recording… Press Ctrl+C to stop.")

    # Simplified approach - rely on ErgManager's 5Hz updates like the working script
    # No background poller needed - the working script doesn't use one

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame timestamp from shared clock
            ts_ns_off = now_ns() - start_ns
            ts_iso = ns_to_iso(ts_ns_off, start_wall)

            # Burn-in timestamp (bottom-right) for visual reference
            cv2.putText(frame, ts_iso, (width - 320, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Add frame number and timestamp as text overlay for debugging
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Offset: {ts_ns_off}ns", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            # Encode frame as PNG and send to FFmpeg
            success, buffer = cv2.imencode('.png', frame)
            if success:
                ffmpeg_process.stdin.write(buffer.tobytes())
                ffmpeg_process.stdin.flush()
            
            # Log frame data to CSV
            frame_writer.writerow({
                "frame_idx": frame_idx, 
                "ts_ns": ts_ns_off, 
                "ts_iso": ts_iso,
                "frame_file": f"frame_{frame_idx:06d}.png"  # Reference to frame file
            })
            f_frames.flush()
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
                    'forceplot_current_json': item['forceplot_current_json'],
                    'forceplot_complete_json': item['forceplot_complete_json'],
                }); f_pm5.flush()

                raw_writer.writerow({
                    'ts_ns': int(item['ts_ns']),
                    'ts_iso': item['ts_iso'],
                    'raw_json': item['raw_json'],
                }); f_raw.flush()

            time.sleep(max(0.0, (1.0 / fps) - 0.001))
    except KeyboardInterrupt:
        print("\n⏹️  Stopping…")
    finally:
        stop_evt.set()
        ergman.stop()
        
        # Close camera
        cap.release()
        
        # Close FFmpeg process
        if ffmpeg_process:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            print("🎥 FFmpeg encoding completed")
        
        # Close CSV files
        f_frames.close(); f_pm5.close(); f_raw.close()
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up temporary directory: {e}")
        
        print("Saved:")
        print("  Folder:", session_dir)
        print("  Video:", video_path)
        print("  Frames:", frames_csv_path)
        print("  PM5:", pm5_csv_path)
        print("  Raw:", raw_csv_path)

if __name__ == "__main__":
    main()


