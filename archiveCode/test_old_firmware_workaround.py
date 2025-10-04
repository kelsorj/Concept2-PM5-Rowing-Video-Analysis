#!/usr/bin/env python3
"""
Test workarounds for older PM5 firmware (version 155)
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
    print("Please run with: sudo ./rowing_env/bin/python3 test_old_firmware_workaround.py")
    exit(1)

def test_alternative_forceplot_commands():
    """Test alternative approaches for older firmware"""
    print("🔍 Testing workarounds for PM5 firmware version 155...")
    
    try:
        devices = list(pyrow.find())
        if not devices:
            print("❌ No PM5 devices found")
            return
        
        device = devices[0]
        erg = pyrow.PyErg(device)
        
        print("✅ Connected to PM5")
        
        # Test 1: Try different CSAFE command variations
        print("\n🔧 Testing alternative CSAFE commands...")
        
        # Test different buffer sizes
        for buffer_size in [16, 32, 64]:
            try:
                print(f"\n   Testing buffer size: {buffer_size}")
                command = ['CSAFE_PM_GET_FORCEPLOTDATA', buffer_size, 'CSAFE_PM_GET_STROKESTATE']
                results = erg.send(command)
                
                if 'CSAFE_PM_GET_FORCEPLOTDATA' in results:
                    fp_raw = results['CSAFE_PM_GET_FORCEPLOTDATA']
                    print(f"   Raw response: {fp_raw}")
                    
                    if len(fp_raw) > 0:
                        datapoints = fp_raw[0] // 2
                        print(f"   Data points: {datapoints}")
                        if datapoints > 0:
                            force_data = fp_raw[1:(datapoints+1)]
                            print(f"   ✅ Force data: {force_data}")
                        else:
                            print(f"   ⚠️  No data points")
                    else:
                        print(f"   ⚠️  Empty response")
                else:
                    print(f"   ❌ No forceplot response")
                    
            except Exception as e:
                print(f"   ❌ Buffer size {buffer_size} failed: {e}")
        
        # Test 2: Try different command combinations
        print("\n🔧 Testing different command combinations...")
        
        test_commands = [
            ['CSAFE_PM_GET_FORCEPLOTDATA', 32],
            ['CSAFE_PM_GET_FORCEPLOTDATA', 16, 'CSAFE_GETSTATUS_CMD'],
            ['CSAFE_PM_GET_STROKESTATE', 'CSAFE_PM_GET_FORCEPLOTDATA', 32],
        ]
        
        for i, command in enumerate(test_commands):
            try:
                print(f"\n   Test {i+1}: {command}")
                results = erg.send(command)
                print(f"   Results: {results}")
                
                # Look for any forceplot data
                for key, value in results.items():
                    if 'FORCEPLOT' in key:
                        print(f"   ✅ Found forceplot data in {key}: {value}")
                        
            except Exception as e:
                print(f"   ❌ Test {i+1} failed: {e}")
        
        # Test 3: Check if we can get stroke statistics instead
        print("\n🔧 Testing stroke statistics (alternative to forceplot)...")
        try:
            # Try to get stroke stats which might be more reliable on older firmware
            command = ['CSAFE_PM_GET_STROKESTATS']
            results = erg.send(command)
            print(f"   Stroke stats command: {command}")
            print(f"   Stroke stats results: {results}")
            
        except Exception as e:
            print(f"   ❌ Stroke stats failed: {e}")
        
        # Test 4: Check PM5 capabilities
        print("\n🔧 Testing PM5 capabilities...")
        try:
            command = ['CSAFE_GETCAPS_CMD']
            results = erg.send(command)
            print(f"   Capabilities command: {command}")
            print(f"   Capabilities results: {results}")
            
        except Exception as e:
            print(f"   ❌ Capabilities test failed: {e}")
        
        return erg
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def monitor_with_workarounds(erg):
    """Monitor using workaround approaches"""
    if not erg:
        print("❌ No erg available")
        return
    
    print("\n" + "="*60)
    print("WORKAROUND MONITORING TEST")
    print("="*60)
    print("💡 Start rowing now! This will test workarounds for 30 seconds...")
    
    start_time = time.time()
    sample_count = 0
    forceplot_found = False
    
    while time.time() - start_time < 30:
        try:
            # Try multiple approaches
            approaches = [
                ("Standard get_monitor(forceplot=True)", lambda: erg.get_monitor(forceplot=True)),
                ("Direct get_forceplot()", lambda: erg.get_forceplot()),
                ("Small buffer forceplot", lambda: erg.send(['CSAFE_PM_GET_FORCEPLOTDATA', 16, 'CSAFE_PM_GET_STROKESTATE'])),
                ("Large buffer forceplot", lambda: erg.send(['CSAFE_PM_GET_FORCEPLOTDATA', 64, 'CSAFE_PM_GET_STROKESTATE'])),
            ]
            
            for approach_name, approach_func in approaches:
                try:
                    result = approach_func()
                    
                    # Extract forceplot data based on result type
                    if isinstance(result, dict):
                        fp_data = result.get('forceplot', [])
                        if 'CSAFE_PM_GET_FORCEPLOTDATA' in result:
                            fp_raw = result['CSAFE_PM_GET_FORCEPLOTDATA']
                            if len(fp_raw) > 0:
                                datapoints = fp_raw[0] // 2
                                if datapoints > 0:
                                    fp_data = fp_raw[1:(datapoints+1)]
                    else:
                        fp_data = []
                    
                    if fp_data and len(fp_data) > 0:
                        forceplot_found = True
                        print(f"✅ {approach_name}: {len(fp_data)} force points - {fp_data[:5]}")
                        break
                        
                except Exception as e:
                    continue  # Try next approach
            
            if not forceplot_found and sample_count % 10 == 0:
                print(f"⏳ Sample {sample_count}: No forceplot data found with any approach")
            
            sample_count += 1
            time.sleep(0.5)  # 2Hz sampling
            
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            break
    
    print(f"\n📊 Workaround monitoring complete:")
    print(f"   Total samples: {sample_count}")
    print(f"   Forceplot data found: {'YES' if forceplot_found else 'NO'}")
    
    if not forceplot_found:
        print("\n❌ NO WORKAROUNDS FOUND!")
        print("💡 Firmware update is strongly recommended.")
        print("💡 Version 155 (2015) is too old for reliable forceplot support.")

def main():
    print("🔍 OLD FIRMWARE WORKAROUND TEST")
    print("="*60)
    print("Testing workarounds for PM5 firmware version 155")
    
    # Test alternative commands
    erg = test_alternative_forceplot_commands()
    
    if erg:
        print("\n" + "="*60)
        print("Press Enter to continue to workaround monitoring test...")
        input()
        monitor_with_workarounds(erg)
    
    print("\n🎯 Workaround test complete!")
    print("💡 If no workarounds work, firmware update is the best solution.")

if __name__ == "__main__":
    main()
