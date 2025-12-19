import time
import tkinter as tk
from lerobot.motors.dynamixel import DynamixelMotorsBus

# CONFIGURATION
PORT = "/dev/ttyACM0"  # Your robot port
ARM_TYPE = "so101"     # or 'so100'

# Connect to the robot
# Note: We use the raw motor bus for direct control to bypass the 'follower' logic
bus = DynamixelMotorsBus(
    port=PORT,
    motors={
        # Standard SO-101 Motor IDs. Verify if yours differ.
        "shoulder_pan": (1, "xl330-m288"),
        "shoulder_lift": (2, "xl330-m288"),
        "elbow_flex":    (3, "xl330-m288"),
        "wrist_flex":    (4, "xl330-m288"),
        "wrist_roll":    (5, "xl330-m288"),
        "gripper":       (6, "xl330-m288"),
    },
)
bus.connect()

# Create the GUI
window = tk.Tk()
window.title(f"LeRobot {ARM_TYPE} Control")
window.geometry("400x400")

def update_motor(name, value):
    # Convert slider value (0-100) to motor units or degrees if needed
    # For simplicity, we write raw integer positions. 
    # YOU MAY NEED TO ADJUST RANGES based on your calibration!
    bus.write(name, int(value))

sliders = {}
# Range 0-4096 is standard for Dynamixel XL330 (0-360 degrees)
# WARNING: Start with a safe range (e.g., 1500-2500) to avoid hitting hard stops
safe_min = 1500
safe_max = 2500

for motor_name in bus.motors.keys():
    frame = tk.Frame(window)
    frame.pack(pady=5)
    tk.Label(frame, text=motor_name, width=15).pack(side=tk.LEFT)
    
    # Create slider
    current_pos = bus.read(motor_name) # Read start position
    s = tk.Scale(frame, from_=safe_min, to=safe_max, orient=tk.HORIZONTAL, length=200,
                 command=lambda val, m=motor_name: update_motor(m, val))
    
    # Set slider to current position so robot doesn't jump
    if current_pos:
        s.set(current_pos)
    s.pack(side=tk.RIGHT)
    sliders[motor_name] = s

window.mainloop()
