#!/usr/bin/env python3
"""
Synchronized USB (PM5) + Video Capture with Shared Timestamps
- Captures Concept2 PM5 data over USB
- Captures FaceTime camera video frames
- Uses a single high-resolution monotonic clock for BOTH streams
- Writes:
  * video file (mp4)
  * per-frame CSV with frame_idx, ts_ns, ts_iso
  * PM5 parsed CSV (superset of fields)

Run (macOS, requires sudo for USB HID):
  sudo ./rowing_env/bin/python3 synchronized_usb_video_capture.py --camera 0 --fps 30
"""

import os, csv, time, json, datetime, threading, queue, argparse
import numpy as np
import cv2

# USB (requires sudo)
try:
    import usb.core, usb.util
    from usb.backend import libusb1
    from pyrow import pyrow
except Exception as e:
    print("WARNING: USB libs not available in this environment.")
    usb = None
    libusb1 = None
    pyrow = None


def now_ns() -> int:
    return time.perf_counter_ns()

def ns_to_iso(ts_ns: int, base_wall: datetime.datetime) -> str:
    # Convert monotonic offset to wall-clock by anchoring at start
    return (base_wall + datetime.timedelta(microseconds=ts_ns / 1000)).isoformat(timespec="milliseconds")


def make_libusb_backend():
    CANDIDATES = [
        "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
        "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
    ]
    lib_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if not lib_path:
        raise SystemExit("libusb not found. Run: brew install libusb")
    return libusb1.get_backend(find_library=lambda _: lib_path)


class PM5Worker(threading.Thread):
    def __init__(self, backend, data_q, stop_evt):
        super().__init__(daemon=True)
        self.backend = backend
        self.data_q = data_q
        self.stop_evt = stop_evt
        self.erg = None
        self._printed_error = False

    def run(self):
        try:
            dev = usb.core.find(idVendor=0x17A4, backend=self.backend)
            if dev is None:
                raise SystemExit("No Concept2 PM detected over USB.")
            self.erg = pyrow.PyErg(dev)
            try:
                info = self.erg.get_erg()
                print("✓ PM5 initialized:", info)
            except Exception as e:
                print("⚠️  PM5 connected but get_erg() failed:", e)
        except Exception as e:
            print(f"❌ PM5 init failed: {e}")
            return

        while not self.stop_evt.is_set():
            try:
                m = self.erg.get_monitor()
                fp = []
                try:
                    v = self.erg.get_forceplot()
                    if v:
                        fp = v
                except Exception:
                    fp = []
                self.data_q.put({
                    "ts_ns": now_ns(),
                    "monitor": m,
                    "forceplot": fp,
                })
            except Exception as e:
                if not self._printed_error:
                    print("⚠️  PM5 read error (will continue):", e)
                    self._printed_error = True
            time.sleep(0.2)  # ~5 Hz


def main():
    parser = argparse.ArgumentParser(description="Synchronized USB+Video capture")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (FaceTime usually 0)")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS")
    parser.add_argument("--out-prefix", default=datetime.datetime.now().strftime("capture_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # Prepare outputs
    video_name = f"{args.out_prefix}.mp4"
    frame_csv = f"{args.out_prefix}_frames.csv"
    pm5_csv   = f"{args.out_prefix}_pm5.csv"

    # Shared clock anchor (wall-clock reference)
    start_wall = datetime.datetime.now()
    start_ns = now_ns()

    # OpenCV video
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # CSV writers
    f_frames = open(frame_csv, "w", newline="")
    fw = csv.DictWriter(f_frames, fieldnames=["frame_idx","ts_ns","ts_iso"])
    fw.writeheader()

    f_pm5 = open(pm5_csv, "w", newline="")
    pm_fields = ["ts_ns","ts_iso","elapsed_s","distance","spm","power","pace","calhr","calories","forceplot_json"]
    pmw = csv.DictWriter(f_pm5, fieldnames=pm_fields)
    pmw.writeheader()

    # Start PM5 thread
    data_q = queue.Queue(maxsize=1000)
    stop_evt = threading.Event()
    backend = None
    if usb is not None:
        try:
            backend = make_libusb_backend()
        except SystemExit as e:
            print(e)
            backend = None
    if os.geteuid() != 0:
        print("WARNING: Not running as root; PM5 access will likely fail on macOS.")
    if backend is not None and pyrow is not None:
        pm5 = PM5Worker(backend, data_q, stop_evt)
        pm5.start()
    else:
        print("⚠️  PM5 backend unavailable; only video frames will be recorded.")

    print("Recording… Press Ctrl+C to stop.")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_curr_ns = now_ns() - start_ns
            ts_iso = ns_to_iso(ts_curr_ns, start_wall)

            # Overlay timestamp on video (same clock used for PM5 records)
            cv2.putText(frame, ts_iso, (width - 320, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out.write(frame)

            fw.writerow({"frame_idx": frame_idx, "ts_ns": ts_curr_ns, "ts_iso": ts_iso}); f_frames.flush()
            frame_idx += 1

            # Drain PM5 queue and write rows with the SAME ts_iso mapping
            while not data_q.empty():
                item = data_q.get_nowait()
                pm_ts_ns = int(item.get("ts_ns", now_ns()) - start_ns)
                pm_ts_iso = ns_to_iso(pm_ts_ns, start_wall)
                m = item.get("monitor", {})
                fp = item.get("forceplot", [])
                pmw.writerow({
                    "ts_ns": pm_ts_ns,
                    "ts_iso": pm_ts_iso,
                    "elapsed_s": m.get("time"),
                    "distance": m.get("distance"),
                    "spm": m.get("spm"),
                    "power": m.get("power"),
                    "pace": m.get("pace"),
                    "calhr": m.get("calhr"),
                    "calories": m.get("calories"),
                    "forceplot_json": json.dumps(fp),
                }); f_pm5.flush()

            # pace capture loop to approximate target FPS
            time.sleep(max(0.0, (1.0 / fps) - 0.001))
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop_evt.set()
        cap.release(); out.release()
        f_frames.close(); f_pm5.close()
        print("Saved:")
        print("  Video:", video_name)
        print("  Frame CSV:", frame_csv)
        print("  PM5 CSV:", pm5_csv)


if __name__ == "__main__":
    main()


