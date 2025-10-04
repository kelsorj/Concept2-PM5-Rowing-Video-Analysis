#!/usr/bin/env python3
"""
Comprehensive diagnostic script to debug forceplot data issues
"""

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
    print("Please run with: sudo ./rowing_env/bin/python3 debug_forceplot_comprehensive.py")
    exit(1)

print("✓ Running with sudo privileges - USB access granted")

def test_direct_pyrow_access():
    """Test direct pyrow access without ErgManager"""
    print("\n" + "="*60)
    print("TESTING DIRECT PYROW ACCESS")
    print("="*60)
    
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return None
        
        print(f"✅ Found {len(devices)} PM5 device(s)")
        
        for i, device in enumerate(devices):
            print(f"\n--- Testing Device {i+1} ---")
            try:
                erg = pyrow.PyErg(device)
                print(f"✅ PyErg created successfully")
                
                # Test basic monitor data
                print("\n📊 Testing get_monitor()...")
                monitor = erg.get_monitor()
                print(f"   Monitor data keys: {list(monitor.keys())}")
                print(f"   Status: {monitor.get('status')}")
                print(f"   Power: {monitor.get('power')}W")
                print(f"   SPM: {monitor.get('spm')}")
                print(f"   Stroke State: {monitor.get('strokestate')}")
                
                # Test monitor with forceplot=True
                print("\n🎯 Testing get_monitor(forceplot=True)...")
                monitor_fp = erg.get_monitor(forceplot=True)
                fp_data = monitor_fp.get('forceplot', [])
                print(f"   Forceplot length: {len(fp_data)}")
                if fp_data:
                    print(f"   Forceplot data: {fp_data[:10]}{'...' if len(fp_data) > 10 else ''}")
                    print(f"   Forceplot range: {min(fp_data)} - {max(fp_data)}")
                else:
                    print("   ⚠️  Forceplot is EMPTY!")
                
                # Test direct get_forceplot()
                print("\n🎯 Testing get_forceplot()...")
                try:
                    fp_result = erg.get_forceplot()
                    print(f"   get_forceplot() result keys: {list(fp_result.keys())}")
                    fp_data = fp_result.get('forceplot', [])
                    print(f"   Forceplot length: {len(fp_data)}")
                    if fp_data:
                        print(f"   Forceplot data: {fp_data[:10]}{'...' if len(fp_data) > 10 else ''}")
                        print(f"   Forceplot range: {min(fp_data)} - {max(fp_data)}")
                    else:
                        print("   ⚠️  Forceplot is EMPTY!")
                except Exception as e:
                    print(f"   ❌ get_forceplot() failed: {e}")
                
                # Test workout data
                print("\n📋 Testing get_workout()...")
                workout = erg.get_workout()
                print(f"   Workout state: {workout.get('state')}")
                print(f"   Workout type: {workout.get('type')}")
                print(f"   Workout status: {workout.get('status')}")
                
                # Test erg data
                print("\n🔧 Testing get_erg()...")
                erg_data = erg.get_erg()
                print(f"   Model: {erg_data.get('model')}")
                print(f"   HW Version: {erg_data.get('hwversion')}")
                print(f"   SW Version: {erg_data.get('swversion')}")
                print(f"   Serial: {erg_data.get('serial')}")
                
                return erg
                
            except Exception as e:
                print(f"❌ Error testing device {i+1}: {e}")
                continue
        
        return None
        
    except Exception as e:
        print(f"❌ Error in direct pyrow access: {e}")
        return None

def test_ergmanager_access():
    """Test ErgManager access"""
    print("\n" + "="*60)
    print("TESTING ERGMANAGER ACCESS")
    print("="*60)
    
    erg_ref = {'erg': None}
    
    def new_erg_callback(erg):
        timestamp = datetime.datetime.now().isoformat()
        print(f"✅ New PM5 detected via ErgManager: {erg}")
        print(f"   Device info: {erg}")
        erg_ref['erg'] = erg
    
    def update_erg_callback(erg):
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
            
            # Test direct access through erg object
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
            
            # Test underlying _pyerg object
            try:
                if hasattr(erg, '_pyerg'):
                    monitor_with_fp = erg._pyerg.get_monitor(forceplot=True)
                    fp_from_monitor = monitor_with_fp.get('forceplot', [])
                    if fp_from_monitor:
                        print(f"   _pyerg.get_monitor(forceplot=True): {len(fp_from_monitor)} samples, range: {min(fp_from_monitor)} - {max(fp_from_monitor)}")
                    else:
                        print("   _pyerg.get_monitor(forceplot=True): Empty")
            except Exception as e:
                print(f"   _pyerg.get_monitor(forceplot=True) failed: {e}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"⚠️  Error processing erg data: {e}")
    
    try:
        print("🔍 Starting ErgManager...")
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
        
        return erg_ref.get('erg')
        
    except Exception as e:
        print(f"❌ Failed to start ErgManager: {e}")
        return None

def test_continuous_monitoring(erg):
    """Test continuous monitoring with different approaches"""
    if not erg:
        print("❌ No erg available for continuous monitoring")
        return
    
    print("\n" + "="*60)
    print("CONTINUOUS MONITORING TEST")
    print("="*60)
    print("💡 Start rowing now! This will monitor for 30 seconds...")
    
    start_time = time.time()
    sample_count = 0
    forceplot_samples = 0
    
    while time.time() - start_time < 30:
        try:
            # Test 1: get_monitor(forceplot=True)
            monitor = erg.get_monitor(forceplot=True)
            fp_data = monitor.get('forceplot', [])
            strokestate = monitor.get('strokestate', '')
            power = monitor.get('power', 0)
            
            if fp_data:
                forceplot_samples += 1
                print(f"✅ Sample {sample_count}: {len(fp_data)} force points, state={strokestate}, power={power}W")
                print(f"   Data: {fp_data[:5]}{'...' if len(fp_data) > 5 else ''}")
            else:
                if sample_count % 10 == 0:  # Print every 10th empty sample
                    print(f"⏳ Sample {sample_count}: No forceplot data, state={strokestate}, power={power}W")
            
            # Test 2: get_forceplot()
            try:
                fp_result = erg.get_forceplot()
                fp_data2 = fp_result.get('forceplot', [])
                if fp_data2 and len(fp_data2) > 0:
                    print(f"🎯 Direct get_forceplot(): {len(fp_data2)} points")
            except Exception as e:
                pass  # Silent fail
            
            sample_count += 1
            time.sleep(0.2)  # 5Hz sampling
            
        except Exception as e:
            print(f"❌ Error in continuous monitoring: {e}")
            break
    
    print(f"\n📊 Monitoring complete:")
    print(f"   Total samples: {sample_count}")
    print(f"   Samples with forceplot: {forceplot_samples}")
    print(f"   Success rate: {forceplot_samples/sample_count*100:.1f}%" if sample_count > 0 else "   Success rate: 0%")

def main():
    print("🔍 COMPREHENSIVE FORCEPLOT DIAGNOSTIC")
    print("="*60)
    print("This script will test multiple approaches to get forceplot data")
    print("Make sure your PM5 is connected and ready to row!")
    
    # Test 1: Direct pyrow access
    erg = test_direct_pyrow_access()
    
    # Test 2: ErgManager access
    print("\n" + "="*60)
    print("Press Enter to continue to ErgManager test...")
    input()
    ergmanager_erg = test_ergmanager_access()
    
    # Test 3: Continuous monitoring
    if erg:
        print("\n" + "="*60)
        print("Press Enter to continue to continuous monitoring test...")
        input()
        test_continuous_monitoring(erg)
    
    print("\n🎯 Diagnostic complete!")
    print("Check the output above to understand why forceplot data might be empty.")

if __name__ == "__main__":
    main()
