#!/bin/bash
# Simple script to start force data capture with sudo
# This is the only way to get complete force curves on macOS

echo "🚣‍♂️ Starting Concept2 PM5 Force Data Capture"
echo "=============================================="
echo ""
echo "⚠️  This requires sudo for USB access on macOS"
echo "   Force data is only available via USB connection"
echo ""

# Check if PM5 is connected
if ! system_profiler SPUSBDataType | grep -q "Concept2"; then
    echo "❌ No Concept2 PM5 detected via USB"
    echo "   Please connect PM5 via USB cable (square USB-B port)"
    echo "   Make sure PM5 is powered on and awake"
    exit 1
fi

echo "✅ Concept2 PM5 detected via USB"
echo ""

# Run the capture script with sudo
echo "🔧 Starting force data capture..."
echo "   You'll be prompted for your password once"
echo ""

sudo ./rowing_env/bin/python3 py3row_usb_capture.py

echo ""
echo "🎉 Force data capture completed!"
echo "   Check the generated CSV files for complete force curves"
