import time
import tkinter as tk
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
# Import the config class
from lerobot.common.robot_devices.motors.configs import DynamixelMotorConfig

# CONFIGURATION
PORT = "/dev/ttyACM0"
ARM_TYPE = "so101"

# Connect to the robot
# We must wrap the motor info in DynamixelMotorConfig objects
bus = DynamixelMotorsBus(
    port=PORT,
    motors={
        "shoulder_pan": DynamixelMotorConfig(id=1, model="xl330-m288"),
        "shoulder_lift": DynamixelMotorConfig(id=2, model="xl330-m288"),
        "elbow_flex":    DynamixelMotorConfig(id=3, model="xl330-m288"),
        "wrist_flex":    DynamixelMotorConfig(id=4, model="xl330-m288"),
        "wrist_roll":    DynamixelMotorConfig(id=5, model="xl330-m288"),
        "gripper":       DynamixelMotorConfig(id=6, model="xl330-m288"),
    },
)
bus.connect()

# Create the GUI
window = tk.Tk()
window.title(f"LeRobot {ARM_TYPE} Control")
window.geometry("400x500") # Made slightly taller

def update_motor(name, value):
    bus.write(name, int(value))

sliders = {}
# Safe range for testing (approx middle of range)
safe_min = 1500
safe_max = 2500

for motor_name in bus.motors.keys():
    frame = tk.Frame(window)
    frame.pack(pady=5)
    
    # Label
    tk.Label(frame, text=motor_name, width=15).pack(side=tk.LEFT)
    
    # Read start position
    current_pos = bus.read(motor_name)
    
    # Create slider
    s = tk.Scale(frame, from_=safe_min, to=safe_max, orient=tk.HORIZONTAL, length=200,
                 command=lambda val, m=motor_name: update_motor(m, val))
    
    # Set slider to current position
    if current_pos:
        s.set(current_pos)
        
    s.pack(side=tk.RIGHT)
    sliders[motor_name] = s

window.mainloop()
