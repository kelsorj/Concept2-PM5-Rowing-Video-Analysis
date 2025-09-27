# rowing.py
# pip install pyusb git+https://github.com/droogmic/Py3Row.git
import os, csv, time, json, datetime, usb.core, usb.util
from usb.backend import libusb1
from pyrow import pyrow

# Locate Homebrew libusb and create backend
CANDIDATES = [
    "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",  # Apple Silicon
    "/usr/local/opt/libusb/lib/libusb-1.0.dylib",     # Intel mac
]
lib_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not lib_path:
    raise SystemExit("libusb not found. Run: brew install libusb")
backend = libusb1.get_backend(find_library=lambda _: lib_path)

# Find PM (Concept2 vendor ID)
C2_VENDOR_ID = 0x17A4
dev = usb.core.find(idVendor=C2_VENDOR_ID, backend=backend)
if dev is None:
    raise SystemExit("No Concept2 PM detected over USB. Use the square USB‑B port and wake the PM5.")

erg = pyrow.PyErg(dev)
print("Device info:", erg.get_erg())

# Filenames
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_mon = f"concept2_monitor_{ts}.csv"
log_fp  = f"concept2_forceplot_{ts}.csv"  # optional

# Open CSVs
mon_fields = ["timestamp","status","time_s","distance_m","spm","watts","pace_s_per_500m","cal_hr","cal_total","hr"]
fp_fields  = ["timestamp","sample_index","force_curve_json"]

with open(log_mon, "w", newline="") as fmon, open(log_fp, "w", newline="") as ffp:
    mon_w = csv.DictWriter(fmon, fieldnames=mon_fields); mon_w.writeheader()
    fp_w  = csv.DictWriter(ffp,  fieldnames=fp_fields);  fp_w.writeheader()

    print("Logging… start a workout (e.g., Just Row). Press Ctrl+C to stop.")
    sample_idx = 0
    try:
        while True:
            m = erg.get_monitor()  # live metrics
            row = {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "status": m.get("status"),
                "time_s": m.get("time"),
                "distance_m": m.get("distance"),
                "spm": m.get("spm"),
                "watts": m.get("power"),
                "pace_s_per_500m": m.get("pace"),
                "cal_hr": m.get("calhr"),
                "cal_total": m.get("calories"),
                "hr": m.get("heartrate"),
            }
            mon_w.writerow(row); fmon.flush()
            print(row)

            # Optional: force curve (array of ints). Safe to call every loop.
            try:
                fp = erg.get_forceplot()  # returns a list (often 32–64 points)
                if fp:
                    fp_w.writerow({
                        "timestamp": row["timestamp"],
                        "sample_index": sample_idx,
                        "force_curve_json": json.dumps(fp),
                    })
                    ffp.flush()
                    sample_idx += 1
            except Exception:
                # Some PM firmware only returns force curve during drive; ignore errors
                pass

            time.sleep(0.2)  # ~5 Hz
    except KeyboardInterrupt:
        print("\nStopping…")

# Cleanup
usb.util.dispose_resources(dev)
print("Saved monitor to:", log_mon)
print("Saved force curves to:", log_fp)
