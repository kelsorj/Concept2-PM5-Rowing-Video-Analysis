import os, usb.core
from usb.backend import libusb1

candidates = [
    "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",  # Apple Silicon
    "/usr/local/opt/libusb/lib/libusb-1.0.dylib",     # Intel mac
]
lib_path = next((p for p in candidates if os.path.exists(p)), None)
backend = libusb1.get_backend(find_library=(lambda _: lib_path) if lib_path else None)
print("Backend:", backend)

dev = usb.core.find(idVendor=0x17A4, backend=backend)  # Concept2 vendor ID
print("Concept2 device found?" , bool(dev))
