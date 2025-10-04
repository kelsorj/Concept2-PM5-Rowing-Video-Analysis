#!/usr/bin/env python3
"""
Test different workout modes to see which provides forceplot data
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
    print("Please run with: sudo ./rowing_env/bin/python3 test_forceplot_workout_modes.py")
    exit(1)

def test_workout_modes():
    """Test different workout modes"""
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return None
        
        device = devices[0]
        erg = pyrow.PyErg(device)
        
        print("🔍 Testing different workout modes...")
        
        # Test current state
        print("\n📊 Current state:")
        monitor = erg.get_monitor(forceplot=True)
        workout = erg.get_workout()
        print(f"   Workout state: {workout.get('state')}")
        print(f"   Workout type: {workout.get('type')}")
        print(f"   Status: {monitor.get('status')}")
        print(f"   Forceplot: {len(monitor.get('forceplot', []))} points")
        
        # Try to set different workout types
        workout_types = [
            (0, "Just Row"),
            (1, "Fixed Distance"),
            (2, "Fixed Time"),
            (3, "Fixed Calorie"),
            (4, "Fixed Watt-minutes"),
            (5, "Custom"),
        ]
        
        for workout_type, name in workout_types:
            print(f"\n🎯 Testing workout type {workout_type} ({name}):")
            try:
                # Try to set the workout type
                if workout_type == 0:  # Just Row
                    erg.set_workout()
                elif workout_type == 1:  # Fixed Distance
                    erg.set_workout(distance=500)  # 500m
                elif workout_type == 2:  # Fixed Time
                    erg.set_workout(workout_time=[0, 2, 0])  # 2 minutes
                
                # Wait a moment for the change to take effect
                time.sleep(1)
                
                # Check the new state
                monitor = erg.get_monitor(forceplot=True)
                workout = erg.get_workout()
                fp_data = monitor.get('forceplot', [])
                
                print(f"   New workout state: {workout.get('state')}")
                print(f"   New workout type: {workout.get('type')}")
                print(f"   Forceplot: {len(fp_data)} points")
                if fp_data:
                    print(f"   ✅ Forceplot data: {fp_data[:5]}")
                
            except Exception as e:
                print(f"   ❌ Error setting workout type {workout_type}: {e}")
        
        return erg
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_while_rowing(erg):
    """Monitor forceplot data while user is rowing"""
    if not erg:
        return
    
    print("\n" + "="*60)
    print("ROWING MONITORING TEST")
    print("="*60)
    print("💡 Start rowing now! This will monitor for 30 seconds...")
    print("💡 Make sure you're actively pulling on the handle!")
    
    start_time = time.time()
    sample_count = 0
    forceplot_found = False
    
    while time.time() - start_time < 30:
        try:
            monitor = erg.get_monitor(forceplot=True)
            workout = erg.get_workout()
            
            fp_data = monitor.get('forceplot', [])
            strokestate = monitor.get('strokestate', '')
            power = monitor.get('power', 0)
            spm = monitor.get('spm', 0)
            workout_state = workout.get('state', '')
            workout_type = workout.get('type', '')
            
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
    
    print(f"\n📊 Monitoring complete:")
    print(f"   Total samples: {sample_count}")
    print(f"   Forceplot data found: {'YES' if forceplot_found else 'NO'}")
    
    if not forceplot_found:
        print("\n❌ NO FORCEPLOT DATA FOUND!")
        print("💡 This suggests the PM5 firmware may not support forceplot data")
        print("💡 Or the PM5 needs to be in a very specific state")

def main():
    print("🔍 FORCEPLOT WORKOUT MODE TESTER")
    print("="*60)
    
    # Test different workout modes
    erg = test_workout_modes()
    
    if erg:
        print("\n" + "="*60)
        print("Press Enter to continue to rowing monitoring test...")
        input()
        monitor_while_rowing(erg)
    
    print("\n🎯 Workout mode test complete!")

if __name__ == "__main__":
    main()
