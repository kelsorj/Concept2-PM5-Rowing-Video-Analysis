#!/bin/bash
# Script to start USB rowing data capture with sudo permissions

echo "Starting USB rowing data capture..."
echo "You may be prompted for your sudo password for USB access."
echo ""

# Activate virtual environment
source rowing_env/bin/activate

# Run the USB script with sudo (preserving environment)
echo "Requesting sudo access for USB device..."
sudo -E rowing_env/bin/python3 enhanced_usb_c2.py

echo ""
echo "USB capture completed."
