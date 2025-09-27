# sanity.py
import os, usb.core, usb.util
from usb.backend import libusb1
from pyrow import pyrow

# Homebrew libusb
CANDIDATES = [
    "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
    "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
]
lib_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if not lib_path:
    raise SystemExit("libusb not found. Run: brew install libusb")

backend = libusb1.get_backend(find_library=lambda _: lib_path)

# Concept2 vendor
C2_VENDOR_ID = 0x17A4
dev = usb.core.find(idVendor=C2_VENDOR_ID, backend=backend)
if dev is None:
    raise SystemExit("No Concept2 PM detected. Use the USB‑B port and wake the PM5.")

erg = pyrow.PyErg(dev)
print("erg info:", erg.get_erg())
print("monitor:", erg.get_monitor())

# Cleanup: release the USB interface/resources (PyErg has no .close())
usb.util.dispose_resources(dev)
