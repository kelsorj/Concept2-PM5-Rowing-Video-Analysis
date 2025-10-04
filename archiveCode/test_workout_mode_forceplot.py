#!/usr/bin/env python3
"""
Test different workout modes to see which one provides forceplot data
"""

import sys
import os
import time
import datetime

# Add Py3Row to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Py3Row'))

from pyrow import pyrow
from pyrow.ergmanager import ErgManager

# Check if we have sudo access
if os.geteuid() != 0:
    print("ERROR: This script requires sudo privileges to access USB devices on macOS.")
    print("Please run with: sudo ./rowing_env/bin/python3 test_workout_mode_forceplot.py")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

def test_workout_modes():
    """Test different workout modes to see which provides forceplot data"""
    print("🔍 Testing different workout modes for forceplot data...")
    
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return
        
        device = devices[0]
        erg = pyrow.PyErg(device)
        
        print("✅ Connected to PM5")
        
        # Test current state
        print("\n📊 Current PM5 state:")
        monitor = erg.get_monitor()
        workout = erg.get_workout()
        print(f"   Status: {monitor.get('status')}")
        print(f"   Workout state: {workout.get('state')}")
        print(f"   Workout type: {workout.get('type')}")
        print(f"   Power: {monitor.get('power')}W")
        print(f"   SPM: {monitor.get('spm')}")
        
        # Test forceplot in current state
        print("\n🎯 Testing forceplot in current state:")
        monitor_fp = erg.get_monitor(forceplot=True)
        fp_data = monitor_fp.get('forceplot', [])
        print(f"   Forceplot length: {len(fp_data)}")
        if fp_data:
            print(f"   ✅ Forceplot data: {fp_data[:5]}")
        else:
            print("   ❌ No forceplot data")
        
        return erg
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_with_different_modes(erg):
    """Monitor forceplot data while trying different workout modes"""
    if not erg:
        print("❌ No erg available")
        return
    
    print("\n" + "="*60)
    print("WORKOUT MODE TESTING")
    print("="*60)
    print("💡 This will test forceplot data in different PM5 states")
    print("💡 Try these steps on your PM5:")
    print("   1. Start with PM5 in 'Ready' state")
    print("   2. Start a 'Just Row' workout")
    print("   3. Try other workout modes if needed")
    print("💡 The script will monitor for 60 seconds...")
    
    start_time = time.time()
    sample_count = 0
    forceplot_found = False
    last_workout_state = None
    last_workout_type = None
    
    while time.time() - start_time < 60:
        try:
            # Get current state
            monitor = erg.get_monitor(forceplot=True)
            workout = erg.get_workout()
            
            fp_data = monitor.get('forceplot', [])
            strokestate = monitor.get('strokestate', '')
            power = monitor.get('power', 0)
            spm = monitor.get('spm', 0)
            workout_state = workout.get('state', '')
            workout_type = workout.get('type', '')
            
            # Check if state changed
            if workout_state != last_workout_state or workout_type != last_workout_type:
                print(f"\n🔄 State change detected:")
                print(f"   Workout state: {last_workout_state} → {workout_state}")
                print(f"   Workout type: {last_workout_type} → {workout_type}")
                last_workout_state = workout_state
                last_workout_type = workout_type
            
            # Check for forceplot data
            if fp_data and len(fp_data) > 0:
                forceplot_found = True
                print(f"✅ Sample {sample_count}: FORCEPLOT DATA FOUND!")
                print(f"   Workout state: {workout_state}, Type: {workout_type}")
                print(f"   Stroke state: {strokestate}, Power: {power}W, SPM: {spm}")
                print(f"   Forceplot: {len(fp_data)} points - {fp_data[:5]}")
            else:
                if sample_count % 10 == 0:  # Print every 10th sample
                    print(f"⏳ Sample {sample_count}: No forceplot, State: {workout_state}, Type: {workout_type}, Power: {power}W")
            
            sample_count += 1
            time.sleep(0.5)  # 2Hz sampling
            
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            break
    
    print(f"\n📊 Workout mode testing complete:")
    print(f"   Total samples: {sample_count}")
    print(f"   Forceplot data found: {'YES' if forceplot_found else 'NO'}")
    
    if not forceplot_found:
        print("\n❌ NO FORCEPLOT DATA FOUND IN ANY MODE!")
        print("💡 Possible issues:")
        print("   - PM5 may need to be in a specific workout mode")
        print("   - Forceplot data may only be available during very specific conditions")
        print("   - Try starting a 'Just Row' workout and rowing actively")
        print("   - Check if PM5 firmware needs updating")

def test_direct_forceplot_calls(erg):
    """Test direct forceplot calls with different approaches"""
    if not erg:
        return
    
    print("\n" + "="*60)
    print("DIRECT FORCEPLOT CALL TESTING")
    print("="*60)
    print("💡 Testing different approaches to get forceplot data...")
    
    approaches = [
        ("get_monitor(forceplot=True)", lambda: erg.get_monitor(forceplot=True)),
        ("get_forceplot()", lambda: erg.get_forceplot()),
        ("Raw CSAFE command", lambda: erg.send(['CSAFE_PM_GET_FORCEPLOTDATA', 32, 'CSAFE_PM_GET_STROKESTATE'])),
    ]
    
    for approach_name, approach_func in approaches:
        print(f"\n🔧 Testing: {approach_name}")
        try:
            result = approach_func()
            
            if isinstance(result, dict):
                fp_data = result.get('forceplot', [])
                if 'CSAFE_PM_GET_FORCEPLOTDATA' in result:
                    fp_raw = result['CSAFE_PM_GET_FORCEPLOTDATA']
                    if len(fp_raw) > 0:
                        datapoints = fp_raw[0] // 2
                        if datapoints > 0:
                            fp_data = fp_raw[1:(datapoints+1)]
                
                if fp_data and len(fp_data) > 0:
                    print(f"   ✅ SUCCESS: {len(fp_data)} force points - {fp_data[:5]}")
                else:
                    print(f"   ❌ No forceplot data")
            else:
                print(f"   ❌ Unexpected result type: {type(result)}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    print("🔍 WORKOUT MODE FORCEPLOT TEST")
    print("="*60)
    print("This script will test forceplot data in different PM5 workout modes")
    
    # Test current state
    erg = test_workout_modes()
    
    if erg:
        # Test direct calls
        test_direct_forceplot_calls(erg)
        
        print("\n" + "="*60)
        print("Press Enter to continue to workout mode monitoring...")
        input()
        monitor_with_different_modes(erg)
    
    print("\n🎯 Workout mode test complete!")

if __name__ == "__main__":
    main()
