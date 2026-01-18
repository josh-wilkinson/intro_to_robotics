import numpy as np
import ev3_dc as ev3
from rich import print

### A class for controlling the gripper for picking up blocks ###
### Feel free to write you own class instead!
###

class Gripper:
    def __init__(self, port=2):
        self.port = port
        self.open = False

    def move_by(self, degrees):
        move_task = (self.motor.move_by(degrees=degrees))
        move_task.start(thread=True)

    def startup(self, ev3_obj):    
        self.motor = ev3.Motor(self.port, ev3_obj=ev3_obj, protocol="USB")
        print("[green]Connected to gripper")

    def open_gripper(self):
        if self.open:
            return False
        else:
            move_task = (self.motor.move_by(degrees=-70))
            move_task.start(thread=True)
            self.open = True
            return True
    
    def close_gripper(self):
        if not self.open:
            return False
        else:
            move_task = (self.motor.move_by(degrees=70))
            move_task.start(thread=True)
            self.open = False
            return True

