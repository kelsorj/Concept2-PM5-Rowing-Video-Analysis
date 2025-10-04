#!/usr/bin/env python3
# Diagnostic script to check PM5 forceplot data availability

import sys
import os
import json
import time
import datetime

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))

from pyrow import pyrow
from pyrow.ergmanager import ErgManager

# Check if we have sudo access
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 pm5_forceplot_diagnostic.py")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

def new_erg_callback(erg):
    """Called when a new erg is detected"""
    timestamp = datetime.datetime.now().isoformat()
    print(f"✅ New PM5 detected: {erg}")
    print(f"   Device info: {erg}")

def update_erg_callback(erg):
    """Called when erg data is updated"""
    timestamp = datetime.datetime.now().isoformat()

    try:
        monitor_data = erg.data
        strokestate = monitor_data.get('strokestate', '')
        forceplot = monitor_data.get('forceplot', [])
        power = monitor_data.get('power', 0)
        time_elapsed = monitor_data.get('time', 0)
        distance = monitor_data.get('distance', 0)
        spm = monitor_data.get('spm', 0)

        print(f"📊 {timestamp}: State={strokestate}, Power={power}W, SPM={spm}, Dist={distance:.1f}m, Time={time_elapsed:.1f}s")
        print(f"   Forceplot length: {len(forceplot)} samples")

        if forceplot:
            print(f"   Forceplot data: {forceplot[:10]}{'...' if len(forceplot) > 10 else ''}")
            print(f"   Forceplot range: {min(forceplot)} - {max(forceplot)}")
        else:
            print("   ⚠️  Forceplot is EMPTY!")
        # Also try direct access if available
        try:
            if hasattr(erg, 'get_forceplot'):
                direct_fp = erg.get_forceplot()
                if direct_fp and 'forceplot' in direct_fp:
                    fp_data = direct_fp['forceplot']
                    print(f"   Direct forceplot: {len(fp_data)} samples, range: {min(fp_data)} - {max(fp_data)}")
                else:
                    print("   Direct forceplot: None or empty")
        except Exception as e:
            print(f"   Direct forceplot access failed: {e}")

        # Try calling get_monitor with forceplot=True explicitly on the underlying PyErg object
        try:
            monitor_with_fp = erg._pyerg.get_monitor(forceplot=True)
            fp_from_monitor = monitor_with_fp.get('forceplot', [])
            if fp_from_monitor:
                print(f"   _pyerg.get_monitor(forceplot=True): {len(fp_from_monitor)} samples, range: {min(fp_from_monitor)} - {max(fp_from_monitor)}")
            else:
                print("   _pyerg.get_monitor(forceplot=True): Empty")
        except Exception as e:
            print(f"   _pyerg.get_monitor(forceplot=True) failed: {e}")

        # Also check if ErgManager is configured correctly
        print(f"   ErgManager update_rate: {erg.rate}")
        print(f"   Erg data keys: {list(monitor_data.keys())}")

        print("-" * 80)

    except Exception as e:
        print(f"⚠️  Error processing erg data: {e}")

try:
    print("🔍 Starting PM5 forceplot diagnostic...")
    print("💡 Make sure PM5 is in 'Just Row' mode and you start rowing")
    print("⏳ Scanning for PM5 devices...")

    ergman = ErgManager(pyrow,
                       add_callback=new_erg_callback,
                       update_callback=update_erg_callback,
                       check_rate=1,
                       update_rate=0.5)  # Faster updates for diagnostics

    print("✅ ErgManager started successfully")
    print("💡 Start rowing on your PM5 and watch for forceplot data...")
    print("Press Ctrl+C to stop diagnostic")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Diagnostic stopped by user")

    ergman.stop()
    print("✅ ErgManager stopped")

except Exception as e:
    print(f"❌ Failed to start diagnostic: {e}")
    exit(1)

print("🎯 Diagnostic complete. Check output above for forceplot data.")
