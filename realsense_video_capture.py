#!/usr/bin/env python3
"""
RealSense Video Capture Script
Requirements:
    pip install pyrealsense2 opencv-python numpy
    
Usage:
    - Press SPACEBAR to start/stop recording
    - Press 'q' to quit
    - Video will be saved as 'captured_video.mp4'
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime

def initialize_realsense():
    """Initialize RealSense pipeline and return pipeline with colorizer."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline, rs.colorizer()

def get_realsense_frames(pipeline, colorizer):
    """Get frames from RealSense camera."""
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_data = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())
    depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    return depth_data, depth_colorized, color

def main():
    """Main function for capturing video."""
    pipeline, colorizer = initialize_realsense()
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    is_recording = False
    frames_captured = 0
    
    # Get frame dimensions from first frame
    _, _, first_frame = get_realsense_frames(pipeline, colorizer)
    frame_height, frame_width = first_frame.shape[:2]
    
    print("RealSense Video Capture")
    print("Press SPACEBAR to start/stop recording")
    print("Press 'q' to quit")
    
    try:
        while True:
            _, _, color = get_realsense_frames(pipeline, colorizer)
            
            # Display current status
            status_text = "RECORDING" if is_recording else "IDLE"
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)  # Red for recording, green for idle
            
            display = color.copy()
            cv2.putText(display, f"Status: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(display, f"Frames: {frames_captured}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RealSense Video Capture", display)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar pressed
                if not is_recording:
                    # Start recording
                    is_recording = True
                    frames_captured = 0
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"captured_video_{timestamp}.mp4"
                    out = cv2.VideoWriter(video_path, fourcc, 30, (frame_width, frame_height))
                    print(f"Recording started. Video will be saved to: {video_path}")
                else:
                    # Stop recording
                    is_recording = False
                    if out:
                        out.release()
                    print(f"Recording stopped. Saved {frames_captured} frames.")
            
            elif key == ord('q'):
                # Quit
                if is_recording and out:
                    out.release()
                break
            
            # Write frame if recording
            if is_recording and out:
                out.write(color)
                frames_captured += 1
    
    finally:
        if out:
            out.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
