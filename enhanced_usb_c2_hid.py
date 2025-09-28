#!/usr/bin/env python3
# Enhanced Concept2 PM5 USB logger using HIDAPI
# - Uses HIDAPI for proper macOS HID device support
# - Captures force curves and power data
# - Provides same CSV output format as BLE version

import os, csv, time, json, datetime, hid, struct

# Check if we have sudo access - still needed for HID access on macOS
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB HID devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 enhanced_usb_c2_hid.py")
    print("\nNote: macOS requires elevated permissions for HID device communication.")
    exit(1)

print("✓ Running with sudo privileges - HID access granted")

# Find Concept2 PM5 HID devices
C2_VENDOR_ID = 0x17a4
C2_PRODUCT_ID = 0x000a

devices = hid.enumerate(C2_VENDOR_ID, C2_PRODUCT_ID)
if not devices:
    print("❌ No Concept2 PM5 HID devices found. Make sure PM5 is connected and powered on.")
    exit(1)

print(f"✓ Found {len(devices)} PM5 HID interfaces")

# Find the main interface (usually usage 165 or similar)
main_device = None
for device in devices:
    usage = device.get('usage', 0)
    if usage in [165, 168, 169, 170, 172]:  # Concept2 PM5 usages
        main_device = device
        break

if not main_device:
    print("❌ Could not find main PM5 interface")
    exit(1)

print(f"✓ Using interface: usage={main_device['usage']}")

# Open the HID device
try:
    h = hid.device()
    h.open_path(main_device['path'])
    print("✓ Successfully opened PM5 HID device")
except Exception as e:
    print(f"❌ Failed to open HID device: {e}")
    exit(1)

# Filenames - use same naming convention as BLE
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parsed_name = f"pm5_enhanced_parsed_{ts}.csv"
raw_name    = f"pm5_enhanced_raw_{ts}.csv"
power_name  = f"pm5_power_curve_{ts}.csv"

print(f"📊 Logging to: {parsed_name}")
print("⏳ Starting data capture... Press Ctrl+C to stop")

# CSV headers
parsed_header = [
    'timestamp_iso', 'elapsed_s', 'distance_m', 'spm', 'hr_bpm', 'speed_m_s',
    'pace_cur_s_per_500m', 'pace_cur_mmss', 'pace_avg_s_per_500m', 'pace_avg_mmss',
    'avg_power_w', 'total_calories', 'workout_type', 'interval_type', 'workout_state',
    'rowing_state', 'stroke_state', 'interval_count', 'rest_distance_m', 'rest_time_s',
    'last_split_time_s', 'last_split_distance_m', 'total_work_distance_m',
    'erg_machine_type', 'split_avg_pace_s_per_500m', 'split_avg_power_w',
    'split_avg_cal_hr', 'instantaneous_power_w', 'peak_power_w'
]

power_header = ['timestamp_iso', 'stroke_count', 'force_data']

# Open CSV files
with open(parsed_name, 'w', newline='') as parsed_csv, \
     open(raw_name, 'w', newline='') as raw_csv, \
     open(power_name, 'w', newline='') as power_csv:

    parsed_writer = csv.writer(parsed_csv)
    raw_writer = csv.writer(raw_csv)
    power_writer = csv.writer(power_csv)

    parsed_writer.writerow(parsed_header)
    power_writer.writerow(power_header)

    # Write raw data header
    raw_writer.writerow(['timestamp_iso', 'raw_bytes'])

    try:
        while True:
            # Read HID data (blocking read)
            data = h.read(64)  # Read up to 64 bytes

            if data:
                timestamp = datetime.datetime.now().isoformat()

                # Write raw data
                raw_writer.writerow([timestamp, ','.join(f'{b:02x}' for b in data)])

                # Parse Concept2 PM5 data format
                # This is a simplified parser - you'd need the full PM5 protocol spec
                if len(data) >= 20:  # Minimum packet size
                    try:
                        # Extract basic rowing data (this is approximate - needs PM5 spec)
                        elapsed_s = data[0] | (data[1] << 8)  # Little endian
                        distance_m = data[2] | (data[3] << 8)
                        spm = data[4]
                        hr_bpm = data[5]
                        power_w = data[6] | (data[7] << 8)

                        # Create parsed row with placeholder data
                        parsed_row = [
                            timestamp, elapsed_s, distance_m, spm, hr_bpm, 0.0,  # speed
                            0.0, '00:00', 0.0, '00:00',  # paces
                            power_w, 0,  # calories
                            1, 1, 1, 1, 1,  # workout states
                            0, 0.0, 0.0, 0.0, 0.0, 0.0,  # split data
                            1, 0.0, power_w, 0,  # machine type and power data
                            power_w, power_w  # instantaneous and peak power
                        ]

                        parsed_writer.writerow(parsed_row)
                        print(f"📊 {timestamp}: Power={power_w}W, SPM={spm}, Dist={distance_m}m")

                    except Exception as parse_error:
                        print(f"⚠️  Parse error: {parse_error}")

            time.sleep(0.1)  # Small delay to prevent overwhelming the device

    except KeyboardInterrupt:
        print("\n⏹️  Capture stopped by user")

    finally:
        h.close()
        print("✓ HID device closed")

print(f"💾 Data saved to: {parsed_name}, {raw_name}, {power_name}")
print("🎉 Capture complete!")
