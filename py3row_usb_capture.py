#!/usr/bin/env python3
# Py3Row USB capture script for Concept2 PM5
# Uses the Py3Row library with ErgManager for robust USB communication

import sys
import os
import csv
import json
import time
import datetime
from pathlib import Path

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))

from pyrow import pyrow
from pyrow.ergmanager import ErgManager

# Check if we have sudo access
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 py3row_usb_capture.py")
    print("\nNote: macOS requires elevated permissions for USB device communication.")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

# Filenames
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
parsed_name = f"pm5_py3row_parsed_{ts}.csv"
raw_name = f"pm5_py3row_raw_{ts}.csv"

print(f"📊 Logging to: {parsed_name}")
print("⏳ Starting Py3Row USB capture... Press Ctrl+C to stop")

# CSV headers
parsed_header = [
    'timestamp_iso', 'elapsed_s', 'distance_m', 'spm', 'hr_bpm', 'speed_m_s',
    'pace_cur_s_per_500m', 'pace_cur_mmss', 'pace_avg_s_per_500m', 'pace_avg_mmss',
    'avg_power_w', 'total_calories', 'workout_type', 'interval_type', 'workout_state',
    'rowing_state', 'stroke_state', 'interval_count', 'rest_distance_m', 'rest_time_s',
    'last_split_time_s', 'last_split_distance_m', 'total_work_distance_m',
    'erg_machine_type', 'split_avg_pace_s_per_500m', 'split_avg_power_w',
    'split_avg_cal_hr', 'instantaneous_power_w', 'peak_power_w', 'forceplot'
]

# Open CSV files
with open(parsed_name, 'w', newline='') as parsed_csv, \
     open(raw_name, 'w', newline='') as raw_csv:

    parsed_writer = csv.writer(parsed_csv)
    raw_writer = csv.writer(raw_csv)

    parsed_writer.writerow(parsed_header)
    raw_writer.writerow(['timestamp_iso', 'raw_data'])

    def new_erg_callback(erg):
        """Called when a new erg is detected"""
        timestamp = datetime.datetime.now().isoformat()
        print(f"✅ New PM5 detected: {erg}")
        raw_writer.writerow([timestamp, f"New erg detected: {erg}"])

    # Global variables to track stroke state and accumulate force data
    current_stroke_force = []
    last_stroke_state = None
    stroke_count = 0

    def update_erg_callback(erg):
        """Called when erg data is updated"""
        global current_stroke_force, last_stroke_state, stroke_count
        timestamp = datetime.datetime.now().isoformat()
        
        try:
            # The erg object has a data attribute with the monitor data
            monitor_data = erg.data
            
            # Log raw data
            raw_writer.writerow([timestamp, json.dumps(monitor_data)])
            
            # Handle force plot aggregation
            current_forceplot = monitor_data.get('forceplot', [])
            current_stroke_state = monitor_data.get('strokestate', '')
            
            # If we have force data, accumulate it
            if current_forceplot:
                current_stroke_force.extend(current_forceplot)
            
            # Check for stroke completion (transition from Drive to Recovery)
            if (last_stroke_state == 'Drive' and current_stroke_state == 'Recovery' and 
                current_stroke_force):
                # Stroke completed - we have the full force curve
                stroke_count += 1
                print(f"🎯 Complete stroke #{stroke_count} captured: {len(current_stroke_force)} force points")
                
                # Store the complete stroke data
                complete_forceplot = current_stroke_force.copy()
                forceplot_str = json.dumps(complete_forceplot)
                
                # Reset for next stroke
                current_stroke_force = []
            else:
                # Still building the stroke or no force data
                forceplot_str = ""
            
            parsed_row = [
                timestamp,
                monitor_data.get('time', 0),
                monitor_data.get('distance', 0),
                monitor_data.get('spm', 0),
                monitor_data.get('heartrate', 0),  # Use 'heartrate' instead of 'hr'
                monitor_data.get('speed', 0),
                monitor_data.get('pace', 0),
                monitor_data.get('pace', 0),  # pace_mmss
                monitor_data.get('pace', 0),  # pace_avg
                monitor_data.get('pace', 0),  # pace_avg_mmss
                monitor_data.get('power', 0),
                monitor_data.get('calories', 0),  # Use 'calories' instead of 'calhr'
                1,  # workout_type
                1,  # interval_type
                1,  # workout_state
                1,  # rowing_state
                1,  # stroke_state
                0,  # interval_count
                0,  # rest_distance_m
                0,  # rest_time_s
                0,  # last_split_time_s
                0,  # last_split_distance_m
                monitor_data.get('distance', 0),  # total_work_distance_m
                1,  # erg_machine_type
                monitor_data.get('pace', 0),  # split_avg_pace
                monitor_data.get('power', 0),  # split_avg_power
                monitor_data.get('calhr', 0),  # split_avg_cal_hr
                monitor_data.get('power', 0),  # instantaneous_power_w
                monitor_data.get('power', 0),  # peak_power_w
                forceplot_str  # forceplot data
            ]
            
            parsed_writer.writerow(parsed_row)
            
            # Update stroke state for next iteration
            last_stroke_state = current_stroke_state
            
            # Console output
            power = monitor_data.get('power', 0)
            distance = monitor_data.get('distance', 0)
            spm = monitor_data.get('spm', 0)
            pace = monitor_data.get('pace', 0)
            forceplot_len = len(current_stroke_force) if current_stroke_force else 0
            
            print(f"📊 {timestamp}: Power={power}W, Dist={distance}m, SPM={spm}, Pace={pace}, State={current_stroke_state}, ForceAccum={forceplot_len}pts")
            
        except Exception as e:
            print(f"⚠️  Error processing erg data: {e}")
            raw_writer.writerow([timestamp, f"Error: {e}"])

    try:
        # Create ErgManager with callbacks
        print("🔍 Scanning for PM5 devices...")
        ergman = ErgManager(pyrow,
                           add_callback=new_erg_callback,
                           update_callback=update_erg_callback,
                           check_rate=1,    # Check for new devices every 1 second
                           update_rate=0.2) # Update data every 200ms (5Hz)
        
        print("✅ Py3Row ErgManager started successfully")
        print("💡 Start rowing on your PM5 to see data...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Capture stopped by user")
        
        # Stop the manager
        ergman.stop()
        print("✅ ErgManager stopped")
        
    except Exception as e:
        print(f"❌ Failed to start Py3Row ErgManager: {e}")
        print("This may be due to USB permission issues or PM5 not being detected.")
        exit(1)

print(f"💾 Data saved to: {parsed_name}, {raw_name}")
print("🎉 Py3Row capture complete!")
