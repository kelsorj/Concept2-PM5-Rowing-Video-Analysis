#!/bin/bash
# Simple script to start rowing data capture with proper environment

echo "🚀 Starting Concept2 PM5 Rowing Data Capture"
echo "=========================================="
echo ""

# Check if we're running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check if PM5 is connected
if ! system_profiler SPUSBDataType | grep -q "Concept2"; then
    echo "❌ No Concept2 PM5 detected. Please connect via USB and try again."
    echo "   Make sure to use the square USB-B port on the PM5."
    exit 1
fi

echo "✅ Concept2 PM5 detected"
echo ""

# Activate the virtual environment and run the script
echo "🔧 Starting capture script..."
echo "Note: You may be prompted for your sudo password."
echo ""

# Provide clear instructions for running with sudo
echo "Please run the following command manually:"
echo ""
echo "    sudo -E ./rowing_env/bin/python3 enhanced_usb_c2.py"
echo ""
echo "Enter your password when prompted."
echo ""
echo "Alternatively, you can run this directly:"
echo "    sudo ./rowing_env/bin/python3 enhanced_usb_c2.py"

echo ""
echo "🎉 Capture complete!"
