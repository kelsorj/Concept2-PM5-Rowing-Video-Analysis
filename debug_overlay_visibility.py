#!/usr/bin/env python3
"""
Debug Overlay Visibility
Check if overlays are being created and positioned correctly
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from datetime import datetime

def create_test_overlay():
    """Create a simple test overlay to verify visibility"""
    # Create a bright red rectangle
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)  # Bright red in BGR
    cv2.putText(img, "TEST OVERLAY", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def create_test_force_plot():
    """Create a simple test force plot"""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Simple force curve
    x = np.arange(20)
    y = np.sin(x * 0.5) * 50 + 50
    ax.plot(x, y, 'lime', linewidth=3)
    ax.plot(10, y[10], 'ro', markersize=10)
    
    ax.set_title('Test Force Plot', color='white', fontsize=12)
    ax.set_xlabel('Position', color='white', fontsize=10)
    ax.set_ylabel('Force', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

def debug_overlay_visibility():
    """Debug overlay visibility by creating a test video"""
    print("🔍 Creating debug overlay visibility test...")
    
    # Open the original video
    video_path = "py3rowcap_20251004_123520/py3rowcap_20251004_123520.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output video
    output_path = "debug_overlay_visibility.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < 100:  # Only process first 100 frames for debugging
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create test overlays
        test_overlay = create_test_overlay()
        test_force_plot = create_test_force_plot()
        
        # Add bright colored overlays in different positions
        # Top-left: Red test overlay
        h1, w1 = test_overlay.shape[:2]
        if w1 <= width and h1 <= height:
            frame[0:h1, 0:w1] = test_overlay
            print(f"Frame {frame_count}: Added red overlay at (0,0) size {w1}x{h1}")
        
        # Top-right: Force plot
        h2, w2 = test_force_plot.shape[:2]
        x2 = width - w2
        if x2 >= 0 and h2 <= height:
            frame[0:h2, x2:x2+w2] = test_force_plot
            print(f"Frame {frame_count}: Added force plot at ({x2},0) size {w2}x{h2}")
        
        # Bottom-left: Green rectangle
        cv2.rectangle(frame, (10, height-100), (200, height-10), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {frame_count}", (20, height-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Bottom-right: Blue rectangle
        cv2.rectangle(frame, (width-200, height-100), (width-10, height-10), (255, 0, 0), -1)
        cv2.putText(frame, "DEBUG", (width-150, height-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % 10 == 0:
            print(f"   Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    print(f"\n🎉 Debug video created: {output_path}")
    print(f"   📊 Processed {frame_count} frames")
    print("   🔍 Check this video to see if overlays are visible")

if __name__ == "__main__":
    debug_overlay_visibility()
