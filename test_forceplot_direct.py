#!/usr/bin/env python3
# Direct PyErg forceplot test - bypasses ErgManager

import sys
import os
import json
import time
import datetime

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))

from pyrow import pyrow

# Check if we have sudo access
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 test_forceplot_direct.py")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

# Find PM5
print("🔍 Finding PM5...")
try:
    devices = list(pyrow.find())
    if not devices:
        print("❌ No PM5 devices found")
        exit(1)
    device = devices[0]
    print(f"✅ Found PM5: {device}")
except Exception as e:
    print(f"❌ Error finding PM5: {e}")
    exit(1)

# Create PyErg object
print("🔧 Initializing PyErg...")
try:
    erg = pyrow.PyErg(device)
    print("✅ PyErg initialized successfully")
except Exception as e:
    print(f"❌ Error initializing PyErg: {e}")
    exit(1)

print("💡 Start rowing on your PM5 now...")
print("Press Ctrl+C to stop testing")

try:
    while True:
        timestamp = datetime.datetime.now().isoformat()

        try:
            # Test regular monitor (should work)
            monitor = erg.get_monitor()
            print(f"📊 Monitor: Power={monitor.get('power', 0)}W, State={monitor.get('strokestate', 'unknown')}, Time={monitor.get('time', 0):.1f}s")

            # Test monitor with forceplot
            monitor_fp = erg.get_monitor(forceplot=True)
            fp_data = monitor_fp.get('forceplot', [])
            print(f"   Forceplot from get_monitor: {len(fp_data)} samples")

            # Test direct forceplot
            try:
                forceplot = erg.get_forceplot()
                fp_direct = forceplot.get('forceplot', [])
                print(f"   Forceplot from get_forceplot: {len(fp_direct)} samples")
            except Exception as e:
                print(f"   get_forceplot failed: {e}")

            if fp_data:
                print(f"   ✅ Forceplot data found! Range: {min(fp_data)} - {max(fp_data)}")
                print(f"   Sample values: {fp_data[:5]}")
            else:
                print("   ❌ No forceplot data from get_monitor")
        except Exception as e:
            print(f"⚠️  Error getting data: {e}")

        print("-" * 60)
        time.sleep(1)  # 1 Hz for testing

except KeyboardInterrupt:
    print("\n⏹️  Test stopped")

print("🎯 Direct forceplot test complete")
