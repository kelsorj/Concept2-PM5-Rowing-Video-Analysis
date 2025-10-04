#!/bin/bash
# Start USB rowing monitor service with sudo

echo "Starting USB Rowing Monitor Service..."
echo "This will run continuously and provide rowing data to the dashboard."
echo "Press Ctrl+C to stop."
echo ""

# Activate virtual environment
source rowing_env/bin/activate

# Run the monitor script with sudo
echo "Requesting sudo access for USB device..."
sudo -E rowing_env/bin/python3 usb_monitor.py
