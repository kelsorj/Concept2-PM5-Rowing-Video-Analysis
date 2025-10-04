#!/usr/bin/env python3
"""
Minimal test script to debug forceplot data issues
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
    print("Please run with: sudo ./rowing_env/bin/python3 test_forceplot_minimal.py")
    exit(1)

def test_raw_csafe_commands():
    """Test raw CSAFE commands to understand what's happening"""
    print("🔍 Testing raw CSAFE commands...")
    
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return
        
        device = devices[0]
        erg = pyrow.PyErg(device)
        
        print("✅ Connected to PM5")
        
        # Test 1: Basic monitor data
        print("\n📊 Testing basic monitor data...")
        monitor = erg.get_monitor()
        print(f"   Status: {monitor.get('status')}")
        print(f"   Power: {monitor.get('power')}W")
        print(f"   SPM: {monitor.get('spm')}")
        print(f"   Stroke State: {monitor.get('strokestate')}")
        
        # Test 2: Monitor with forceplot
        print("\n🎯 Testing monitor with forceplot...")
        monitor_fp = erg.get_monitor(forceplot=True)
        fp_data = monitor_fp.get('forceplot', [])
        print(f"   Forceplot length: {len(fp_data)}")
        print(f"   Forceplot data: {fp_data}")
        
        # Test 3: Direct forceplot
        print("\n🎯 Testing direct forceplot...")
        fp_result = erg.get_forceplot()
        print(f"   get_forceplot() result: {fp_result}")
        
        # Test 4: Raw CSAFE command
        print("\n🔧 Testing raw CSAFE command...")
        try:
            # Send the exact command that get_forceplot uses
            command = ['CSAFE_PM_GET_FORCEPLOTDATA', 32, 'CSAFE_PM_GET_STROKESTATE']
            results = erg.send(command)
            print(f"   Raw CSAFE results: {results}")
            
            # Parse the results manually
            if 'CSAFE_PM_GET_FORCEPLOTDATA' in results:
                fp_raw = results['CSAFE_PM_GET_FORCEPLOTDATA']
                print(f"   Raw forceplot data: {fp_raw}")
                if len(fp_raw) > 0:
                    datapoints = fp_raw[0] // 2
                    print(f"   Data points: {datapoints}")
                    if datapoints > 0:
                        force_data = fp_raw[1:(datapoints+1)]
                        print(f"   Force values: {force_data}")
                    else:
                        print("   ⚠️  No data points returned!")
                else:
                    print("   ⚠️  Empty forceplot response!")
            
            if 'CSAFE_PM_GET_STROKESTATE' in results:
                stroke_state = results['CSAFE_PM_GET_STROKESTATE']
                print(f"   Stroke state: {stroke_state}")
                
        except Exception as e:
            print(f"   ❌ Raw CSAFE command failed: {e}")
        
        # Test 5: Check PM5 capabilities
        print("\n🔧 Testing PM5 capabilities...")
        try:
            erg_data = erg.get_erg()
            print(f"   Model: {erg_data.get('model')}")
            print(f"   HW Version: {erg_data.get('hwversion')}")
            print(f"   SW Version: {erg_data.get('swversion')}")
            print(f"   Serial: {erg_data.get('serial')}")
        except Exception as e:
            print(f"   ❌ Failed to get erg data: {e}")
        
        return erg
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_while_rowing(erg):
    """Monitor forceplot data while user is rowing"""
    if not erg:
        print("❌ No erg available")
        return
    
    print("\n" + "="*60)
    print("ROWING MONITORING TEST")
    print("="*60)
    print("💡 Start rowing now! This will monitor for 20 seconds...")
    print("💡 Make sure you're actively pulling on the handle!")
    
    start_time = time.time()
    sample_count = 0
    forceplot_found = False
    
    while time.time() - start_time < 20:
        try:
            # Test both methods
            monitor_fp = erg.get_monitor(forceplot=True)
            fp_result = erg.get_forceplot()
            
            fp_data1 = monitor_fp.get('forceplot', [])
            fp_data2 = fp_result.get('forceplot', [])
            
            strokestate = monitor_fp.get('strokestate', '')
            power = monitor_fp.get('power', 0)
            
            if fp_data1 or fp_data2:
                forceplot_found = True
                print(f"✅ Sample {sample_count}: FORCEPLOT DATA FOUND!")
                print(f"   Monitor forceplot: {len(fp_data1)} points - {fp_data1[:5] if fp_data1 else 'empty'}")
                print(f"   Direct forceplot: {len(fp_data2)} points - {fp_data2[:5] if fp_data2 else 'empty'}")
                print(f"   State: {strokestate}, Power: {power}W")
            else:
                if sample_count % 5 == 0:  # Print every 5th sample
                    print(f"⏳ Sample {sample_count}: No forceplot, State: {strokestate}, Power: {power}W")
            
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
        print("💡 Possible issues:")
        print("   - PM5 firmware version may not support forceplot")
        print("   - PM5 may need to be in a specific workout mode")
        print("   - Forceplot data may only be available during very specific conditions")
        print("   - Try different workout types (Just Row, Distance, Time, etc.)")

def main():
    print("🔍 MINIMAL FORCEPLOT TEST")
    print("="*60)
    
    # Test raw CSAFE commands
    erg = test_raw_csafe_commands()
    
    if erg:
        print("\n" + "="*60)
        print("Press Enter to continue to rowing monitoring test...")
        input()
        monitor_while_rowing(erg)
    
    print("\n🎯 Test complete!")

if __name__ == "__main__":
    main()
