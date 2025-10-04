#!/usr/bin/env python3
"""
Check PM5 workout state and forceplot availability
"""

import sys
import os
import time

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))

from pyrow import pyrow

# Check if we have sudo access
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 check_pm5_workout_state.py")
    exit(1)

def check_pm5_state():
    """Check current PM5 state and forceplot availability"""
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return None
        
        device = devices[0]
        erg = pyrow.PyErg(device)
        
        print("📊 Current PM5 state:")
        monitor = erg.get_monitor()
        workout = erg.get_workout()
        
        print(f"   Status: {monitor.get('status')}")
        print(f"   Workout state: {workout.get('state')}")
        print(f"   Workout type: {workout.get('type')}")
        print(f"   Power: {monitor.get('power')}W")
        print(f"   SPM: {monitor.get('spm')}")
        print(f"   Stroke state: {monitor.get('strokestate')}")
        
        print("\n🎯 Testing forceplot:")
        monitor_fp = erg.get_monitor(forceplot=True)
        fp_data = monitor_fp.get('forceplot', [])
        print(f"   Forceplot length: {len(fp_data)}")
        if fp_data:
            print(f"   ✅ Forceplot data: {fp_data[:5]}")
        else:
            print("   ❌ No forceplot data")
        
        print("\n🔧 PM5 capabilities:")
        erg_data = erg.get_erg()
        print(f"   Model: {erg_data.get('model')}")
        print(f"   HW Version: {erg_data.get('hwversion')}")
        print(f"   SW Version: {erg_data.get('swversion')}")
        
        return erg
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_workout_changes(erg):
    """Monitor PM5 state changes while user changes workout mode"""
    if not erg:
        return
    
    print("\n" + "="*60)
    print("WORKOUT STATE MONITORING")
    print("="*60)
    print("💡 Try these steps on your PM5:")
    print("   1. Make sure PM5 is in 'Ready' state")
    print("   2. Start a 'Just Row' workout")
    print("   3. Begin rowing actively")
    print("   4. Try other workout modes if needed")
    print("💡 The script will monitor for 60 seconds...")
    
    start_time = time.time()
    last_workout_state = None
    last_workout_type = None
    forceplot_found = False
    
    while time.time() - start_time < 60:
        try:
            monitor = erg.get_monitor(forceplot=True)
            workout = erg.get_workout()
            
            current_workout_state = workout.get('state', '')
            current_workout_type = workout.get('type', '')
            fp_data = monitor.get('forceplot', [])
            strokestate = monitor.get('strokestate', '')
            power = monitor.get('power', 0)
            spm = monitor.get('spm', 0)
            
            # Check for state changes
            if (current_workout_state != last_workout_state or 
                current_workout_type != last_workout_type):
                print(f"\n🔄 State change detected:")
                print(f"   Workout state: {last_workout_state} → {current_workout_state}")
                print(f"   Workout type: {last_workout_type} → {current_workout_type}")
                last_workout_state = current_workout_state
                last_workout_type = current_workout_type
            
            # Check for forceplot data
            if fp_data and len(fp_data) > 0:
                forceplot_found = True
                print(f"✅ FORCEPLOT DATA FOUND!")
                print(f"   Workout state: {current_workout_state}, Type: {current_workout_type}")
                print(f"   Stroke state: {strokestate}, Power: {power}W, SPM: {spm}")
                print(f"   Forceplot: {len(fp_data)} points - {fp_data[:5]}")
            else:
                # Print occasionally when no forceplot data
                if int(time.time()) % 5 == 0:  # Every 5 seconds
                    print(f"⏳ No forceplot, State: {current_workout_state}, Type: {current_workout_type}, Power: {power}W")
            
            time.sleep(0.5)  # 2Hz sampling
            
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            break
    
    print(f"\n📊 Monitoring complete:")
    print(f"   Forceplot data found: {'YES' if forceplot_found else 'NO'}")
    
    if not forceplot_found:
        print("\n❌ NO FORCEPLOT DATA FOUND!")
        print("💡 Possible solutions:")
        print("   - Make sure PM5 is in an active workout (not just 'Ready')")
        print("   - Try 'Just Row' workout mode")
        print("   - Ensure you're actively rowing (not just sitting)")
        print("   - Check if PM5 firmware needs updating")

def main():
    print("🔍 PM5 WORKOUT STATE CHECKER")
    print("="*60)
    
    # Check current state
    erg = check_pm5_state()
    
    if erg:
        print("\n" + "="*60)
        print("Press Enter to continue to workout state monitoring...")
        input()
        monitor_workout_changes(erg)
    
    print("\n🎯 Workout state check complete!")

if __name__ == "__main__":
    main()
