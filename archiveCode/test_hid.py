#!/usr/bin/env python3
# Test HID communication with PM5

import hid

print("Testing HID device access...")

# Find Concept2 PM5
devices = hid.enumerate(0x17a4, 0x000a)
print(f'Found {len(devices)} Concept2 PM5 devices:')

for device in devices:
    print(f'  Path: {device["path"]}')
    print(f'  Product: {device["product_string"]}')
    print(f'  Usage: {device.get("usage", "N/A")}')
    print(f'  Usage Page: {device.get("usage_page", "N/A")}')

    # Try to open the device
    try:
        h = hid.device()
        h.open_path(device['path'])
        print('  ✅ Successfully opened device!')

        # Try to read some data (non-blocking)
        data = h.read(64)
        print(f'  Read {len(data)} bytes: {data[:20] if data else "No data"}')

        h.close()
        print('  ✅ Device communication successful!')

    except Exception as e:
        print(f'  ❌ Failed: {e}')
