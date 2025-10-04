#!/usr/bin/env python3
# USB Rowing Monitor Service
# Runs with sudo privileges and provides rowing data via HTTP API

import os, csv, time, json, datetime, usb.core, usb.util
from usb.backend import libusb1
from pyrow import pyrow
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables
erg = None
current_data = {
    'timestamp': None,
    'elapsed_s': 0,
    'distance_m': 0,
    'spm': 0,
    'power_w': 0,
    'pace_s_per_500m': 0,
    'heart_rate': 0,
    'status': 'disconnected'
}

# Setup USB connection
def setup_usb():
    global erg, current_data

    try:
        # Locate Homebrew libusb
        CANDIDATES = [
            "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
            "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
        ]
        lib_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
        if not lib_path:
            raise SystemExit("libusb not found")

        backend = libusb1.get_backend(find_library=lambda _: lib_path)
        C2_VENDOR_ID = 0x17A4
        dev = usb.core.find(idVendor=C2_VENDOR_ID, backend=backend)

        if dev is None:
            current_data['status'] = 'no_device'
            return False

        erg = pyrow.PyErg(dev)
        print("USB connection established:", erg.get_erg())
        current_data['status'] = 'connected'
        return True

    except Exception as e:
        print(f"USB setup failed: {e}")
        current_data['status'] = 'error'
        return False

@app.route('/api/status')
def get_status():
    return jsonify(current_data)

@app.route('/api/data')
def get_data():
    if erg and current_data['status'] == 'connected':
        try:
            m = erg.get_monitor()
            current_data.update({
                'timestamp': datetime.datetime.now().isoformat(),
                'elapsed_s': m.get('time', 0),
                'distance_m': m.get('distance', 0),
                'spm': m.get('spm', 0),
                'power_w': m.get('power', 0),
                'pace_s_per_500m': m.get('pace', 0),
                'heart_rate': m.get('heartrate', 0)
            })
        except Exception as e:
            current_data['status'] = 'error'
            print(f"Data read error: {e}")

    return jsonify(current_data)

if __name__ == '__main__':
    print("Starting USB Rowing Monitor Service...")
    print("This requires sudo for USB access.")

    if setup_usb():
        print("USB connected. Starting HTTP server on port 3002...")
        app.run(host='127.0.0.1', port=3002, debug=False)
    else:
        print("Failed to connect to USB device. Exiting.")
        exit(1)
