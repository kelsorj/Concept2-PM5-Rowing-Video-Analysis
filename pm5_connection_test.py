#!/usr/bin/env python3
# Simple PM5 connection test (no sudo needed)

import usb.core
import usb.util
from usb.backend import libusb1

print("🔍 Testing PM5 USB connection...")

# Find Homebrew libusb
CANDIDATES = [
    "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",  # Apple Silicon
    "/usr/local/opt/libusb/lib/libusb-1.0.dylib",     # Intel mac
]
lib_path = next((p for p in CANDIDATES if __import__('os').path.exists(p)), None)
if not lib_path:
    print("❌ libusb not found. Run: brew install libusb")
    exit(1)

backend = libusb1.get_backend(find_library=lambda _: lib_path)
C2_VENDOR_ID = 0x17a4

# Find PM5
dev = usb.core.find(idVendor=C2_VENDOR_ID, backend=backend)
if dev is None:
    print("❌ No Concept2 PM5 detected over USB")
    print("💡 Check:")
    print("   - PM5 is powered on")
    print("   - USB cable is connected to square USB-B port")
    print("   - Try a different USB port")
else:
    print(f"✅ Found Concept2 PM5: {dev}")
    print(f"   Vendor ID: 0x{dev.idVendor:04x}")
    print(f"   Product ID: 0x{dev.idProduct:04x}")
    print(f"   Bus: {dev.bus}, Address: {dev.address}")

    # Check if we can claim interface (requires sudo)
    try:
        dev.set_configuration()
        usb.util.claim_interface(dev, 0)
        print("✅ USB interface accessible (sudo working)")
        usb.util.dispose_resources(dev)
    except Exception as e:
        print(f"⚠️  USB interface not accessible: {e}")
        print("   This is expected without sudo - run diagnostic with sudo")

print("🎯 Connection test complete")
