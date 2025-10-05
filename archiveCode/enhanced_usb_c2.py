#!/usr/bin/env python3
# Enhanced Concept2 PM5 USB logger with force curves
# - Connects via USB for complete data access
# - Captures force curves and power data
# - Provides same CSV output format as BLE version

import os, csv, time, json, datetime

# Check if we have sudo access - required for macOS USB HID devices
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 enhanced_usb_c2.py")
    print("\nNote: macOS requires elevated permissions for all USB HID device communication.")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

import usb.core, usb.util
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

try:
    erg = pyrow.PyErg(dev)
    print("✓ Device initialized successfully")
    print("Device info:", erg.get_erg())
except Exception as e:
    print(f"❌ Failed to initialize PM5: {e}")
    print("This may be due to USB permission issues even with sudo.")
    print("Try unplugging and replugging the PM5, then run the script again.")
    exit(1)

# Filenames - use same naming convention as BLE
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parsed_name = f"pm5_enhanced_parsed_{ts}.csv"
raw_name    = f"pm5_enhanced_raw_{ts}.csv"
power_name  = f"pm5_power_curve_{ts}.csv"

# Enhanced parsed fields (same as BLE version)
parsed_fields = [
    "timestamp_iso","elapsed_s","distance_m","spm","hr_bpm","speed_m_s",
    "pace_cur_s_per_500m","pace_cur_mmss","pace_avg_s_per_500m","pace_avg_mmss",
    "avg_power_w","instantaneous_power_w","peak_power_w","total_calories",
    "workout_type","interval_type","workout_state","rowing_state","stroke_state",
    "interval_count","rest_distance_m","rest_time_s",
    "last_split_time_s","last_split_distance_m","total_work_distance_m","erg_machine_type",
    "split_avg_pace_s_per_500m","split_avg_power_w","split_avg_cal_hr",
    "stroke_count","stroke_rate_spm","stroke_distance_m","stroke_time_ms",
    "drive_time_ms","recovery_time_ms","total_strokes","avg_stroke_rate_spm"
]

pf = open(parsed_name, "w", newline="")
pw = csv.DictWriter(pf, fieldnames=parsed_fields); pw.writeheader()

raw_fields = ["ts_ns","source","data"]
rf = open(raw_name, "w", newline="")
rw = csv.DictWriter(rf, fieldnames=raw_fields); rw.writeheader()

power_fields = ["timestamp_iso","elapsed_s","instantaneous_power_w","peak_power_w","stroke_count","force_curve_json"]
powerf = open(power_name, "w", newline="")
powerw = csv.DictWriter(powerf, fieldnames=power_fields); powerw.writeheader()

# High-res timebase
t0 = time.perf_counter_ns()
def ts_ns(): return time.perf_counter_ns() - t0

print("Connected via USB. Start rowing (Just Row mode) and data will be captured.")
print(f"Logging enhanced parsed data → {parsed_name}")
print(f"Logging raw packets → {raw_name}")
print(f"Logging force curves → {power_name}")
print("Press Ctrl+C to stop.")

sample_idx = 0
try:
    while True:
        # Get monitor data
        m = erg.get_monitor()

        # Log raw data
        rw.writerow({
            "ts_ns": ts_ns(),
            "source": "monitor",
            "data": json.dumps(m)
        }); rf.flush()

        # Convert to our standard format
        row = {
            "timestamp_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": m.get("time", 0),
            "distance_m": m.get("distance", 0),
            "spm": m.get("spm", 0),
            "hr_bpm": m.get("heartrate"),
            "speed_m_s": 0,  # Will calculate from pace
            "pace_cur_s_per_500m": m.get("pace"),
            "pace_cur_mmss": None,  # Will be calculated
            "pace_avg_s_per_500m": None,
            "pace_avg_mmss": None,
            "avg_power_w": m.get("power"),  # This is actually current power!
            "instantaneous_power_w": m.get("power"),  # Same as above for USB
            "peak_power_w": None,  # Will try to get from force curve
            "total_calories": m.get("calories"),
            "workout_type": None,
            "interval_type": None,
            "workout_state": "WorkoutRow",  # Assume active rowing
            "rowing_state": "Active",
            "stroke_state": "Unknown",
            "interval_count": 0,
            "rest_distance_m": 0,
            "rest_time_s": 0,
            "last_split_time_s": 0,
            "last_split_distance_m": 0,
            "total_work_distance_m": m.get("distance", 0),
            "erg_machine_type": None,
            "split_avg_pace_s_per_500m": m.get("pace"),
            "split_avg_power_w": m.get("power"),
            "split_avg_cal_hr": m.get("calhr"),
            "stroke_count": sample_idx,
            "stroke_rate_spm": m.get("spm"),
            "stroke_distance_m": 0,
            "stroke_time_ms": 0,
            "drive_time_ms": 0,
            "recovery_time_ms": 0,
            "total_strokes": sample_idx,
            "avg_stroke_rate_spm": m.get("spm")
        }

        # Calculate speed from pace (pace is seconds per 500m)
        if row["pace_cur_s_per_500m"]:
            # Speed in m/s = 500m / pace_seconds
            row["speed_m_s"] = 500.0 / row["pace_cur_s_per_500m"]

        # Calculate pace display strings
        def sec_to_mmss(x):
            if x is None or x == 0: return None
            m = int(x // 60); s = x - 60*m
            return f"{m}:{s:04.1f}"

        if row["pace_cur_s_per_500m"]:
            row["pace_cur_mmss"] = sec_to_mmss(row["pace_cur_s_per_500m"])
        if row["pace_avg_s_per_500m"]:
            row["pace_avg_mmss"] = sec_to_mmss(row["pace_avg_s_per_500m"])

        # Get force curve data
        force_curve = None
        try:
            fp = erg.get_forceplot()
            if fp:
                force_curve = fp
                # Calculate peak power from force curve (rough estimate)
                if fp and len(fp) > 0:
                    # Force curve values are typically 0-1023, convert to approximate watts
                    # This is a rough approximation - real power calculation is complex
                    max_force = max(fp) if fp else 0
                    # Very rough conversion: assume max force * some factor
                    row["peak_power_w"] = int(max_force * 0.3)  # Rough estimate
        except Exception:
            # Force curve may not be available during certain stroke phases
            pass

        # Log force curve data
        if force_curve:
            powerw.writerow({
                "timestamp_iso": row["timestamp_iso"],
                "elapsed_s": row["elapsed_s"],
                "instantaneous_power_w": row["instantaneous_power_w"],
                "peak_power_w": row["peak_power_w"],
                "stroke_count": row["stroke_count"],
                "force_curve_json": json.dumps(force_curve)
            }); powerf.flush()

        # Write parsed data
        pw.writerow({k: row.get(k) for k in parsed_fields}); pf.flush()

        # Console output (similar to BLE version)
        et   = f"{row.get('elapsed_s',0.0):6.1f}s"
        dist = f"{row.get('distance_m',0.0):7.1f}m"
        spm  = f"{row.get('spm','--'):>2}spm"
        hr   = f"{row.get('hr_bpm','--'):>3}bpm"
        pace = row.get("pace_cur_mmss","-:--.-")
        avg_pwr = f"{row.get('avg_power_w','--'):>3}W"
        inst_pwr = f"{row.get('instantaneous_power_w','--'):>3}W"
        peak_pwr = f"{row.get('peak_power_w','--'):>3}W"
        stroke = f"{row.get('stroke_count','--'):>3}"

        print(f"{et} {dist}  {spm}  {hr}  {pace}  Avg:{avg_pwr}  Inst:{inst_pwr}  Peak:{peak_pwr}  Stroke:{stroke}")

        sample_idx += 1
        time.sleep(0.2)  # ~5 Hz updates

except KeyboardInterrupt:
    print("\nStopping…")

# Cleanup
usb.util.dispose_resources(dev)
pf.close(); rf.close(); powerf.close()
print("Saved:")
print("  ", parsed_name)
print("  ", raw_name)
print("  ", power_name)
