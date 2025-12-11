import cv2
import time
import numpy as np
import sys
import jsonpickle
import pickle
from message import Message
from timeit import default_timer as timer

from utils.robot_controller import RobotController

from publisher import Publisher
from utils.keypress_listener import KeypressListener
from rich import print
from utils.utils import load_config

### Note on the main and code in general ###
# This python file contains the main class, which controls all robot behaviour. All top level controls are happening here.
# This file is here to give you something to work with and to base your own implementation on.
# However: You are not limited to our code! If you want to do something differently, feel free to change any part of the code you 
# like. None of the code we give you is mandatory. If you really want to, you can write everything yourself.
###

from enum import Enum
class TaskPart(Enum):
    """
    A helper Enum for the mode we are in.
    """
    Manual = 0
    ### You can add your own Enums here ###

    ###


class Main():
    def __init__(self) -> None:
        """
        
        """

        # load config
        self.config = load_config("config.yaml")

        # instantiate methods
        self.robot = RobotController(self.config)
        self.keypress_listener = KeypressListener()
        self.publisher = Publisher()

        # set default values
        self.DT = self.config.robot.delta_t # delta time in seconds

        self.speed = 0
        self.turn = 0
        self.new_speed = 0
        self.new_turn = 0

        self.manualMode = False
        self.is_running = True

        self.map = None

        self.mode = TaskPart.Manual

        self.run_loop()
        
    def detect_red_blue(self, img):
        """
        Detects red or blue objects in a BGR image.
        Returns: ("red", contour, center) or ("blue", contour, center) or (None, None, None)
        """

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- Red mask (two ranges because red wraps around HSV hue) ---
        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 80])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # --- Blue mask ---
        lower_blue = np.array([100, 150, 60])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        # Find contours for both
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Helper function: return biggest contour and its center
        def get_biggest(contours):
            if len(contours) == 0:
                return None, None
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] == 0:
                return c, None
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return c, center

        c_red, center_red = get_biggest(contours_red)
        c_blue, center_blue = get_biggest(contours_blue)

        # Decide which color is detected (take the one with bigger area)
        area_red = cv2.contourArea(c_red) if c_red is not None else 0
        area_blue = cv2.contourArea(c_blue) if c_blue is not None else 0

        if area_red > 500 and area_red > area_blue:
            return "red", c_red, center_red

        if area_blue > 500 and area_blue > area_red:
            return "blue", c_blue, center_blue

        return None, None, None


    def run_loop(self):
        """
        this loop wraps the methods that use the __enter__ and __close__ functions:
            self.keypress_listener, self.publisher, self.robot
        
        then it calls run()
        """
        print("starting...")

        # control vehicle movement and visualize it
        with self.keypress_listener, self.publisher, self.robot:
            print("starting EKF SLAM...")

            print("READY!")
            print("[green]MODE: Manual")

            count = 0

            while self.is_running:
                time0 = timer()
                self.run(count, time0)

                elapsed_time = timer() - time0
                if elapsed_time <= self.DT:
                    dt = self.DT - elapsed_time
                    time.sleep(dt) # moves while sleeping
                else:
                    print(f"[red]Warning! dt = {elapsed_time}")

                count += 1

            print("*** END PROGRAM ***")

    def run(self, count, time0):
        """
        Were we get the key press, and set the mode accordingly.
        We can use the robot recorder to playback a recording.
        """


        if not self.robot.recorder.playback:
            # read webcam and get distance from aruco markers
            _, raw_img, cam_fps, img_created = self.robot.camera.read() # BGR color

            speed = self.speed
            turn = self.turn
        else:
            cam_fps = 0
            raw_img, speed, turn = next(self.robot.recorder.get_step)

        if raw_img is None:
            print("[red]image is None!")
            return

        draw_img = raw_img.copy()
        data = None
        # Once you have implemented EKF slam, you can use the data to create messages for the viewer
        # data = self.robot.run_ekf_slam(raw_img, draw_img)

        self.parse_keypress()

        if self.mode == TaskPart.Manual:
            self.robot.move(speed, turn)

        ### You can add your own modes and assiciated behaviour here ###
        #if self.mode == TaskPart.Enum:
        #   pass
        ###

        color, contour, center = self.detect_red_blue(draw_img)

        if color is not None:
            # Draw contour
            cv2.drawContours(draw_img, [contour], -1, (0, 255, 0), 2)

            # Draw center
            if center is not None:
                cv2.circle(draw_img, center, 5, (0, 255, 255), -1)

            # Display what was detected
            cv2.putText(draw_img, f"{color.upper()} OBJECT", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        
        # create a message for the viewer
        if not (data is None): 
            msg = Message(
                id = count,
                timestamp = time0,
                start = True,

                landmark_ids = data.landmark_ids,
                landmark_rs = data.landmark_rs,
                landmark_alphas = data.landmark_alphas,
                landmark_positions = data.landmark_positions,

                landmark_estimated_ids = data.landmark_estimated_ids,
                landmark_estimated_positions = data.landmark_estimated_positions,
                landmark_estimated_stdevs = data.landmark_estimated_stdevs,

                robot_position = data.robot_position,
                robot_theta = data.robot_theta,
                robot_stdev = data.robot_stdev,

                text = f"cam fps: {cam_fps}"
            )
        else:
            msg = Message(
                id = count,
                timestamp = time0,
                start = True,

                landmark_ids = [],
                landmark_rs = [],
                landmark_alphas = [],
                landmark_positions = [],

                landmark_estimated_ids = [],
                landmark_estimated_positions = [],
                landmark_estimated_stdevs = [],

                robot_position = 0,
                robot_theta = 0,
                robot_stdev = 0,

                text = f"cam fps: {cam_fps}"
            )

        msg_str = jsonpickle.encode(msg)

        # send message to the viewer
        self.publisher.publish_img(msg_str, draw_img)


    def save_state(self, data):
        ### Your code here ###
        with open("SLAM.pickle", 'wb') as pickle_file:
            pass
        
        pass
        ###

    def load_and_localize(self):
        ### Your code here ###
        with open("SLAM.pickle", 'rb') as f:
            pass

        pass
        ### 

    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()

        turn_step = 40
        speed_step = 5

        if char == "a":
            if self.turn >= 0:
                self.new_turn = self.turn + turn_step
            else:
                self.new_turn = 0
            self.new_turn = min(self.new_turn, 200)
        elif char == "d":
            if self.turn <= 0:
                self.new_turn = self.turn - turn_step
            else:
                self.new_turn = 0
            self.new_turn = max(self.new_turn, -200)
        elif char == "w":
            if self.speed >= 0:
                self.new_speed = self.speed + speed_step
            else:
                self.new_speed = 0
            self.new_speed = min(self.new_speed, 100)
        elif char == "s":
            if self.speed <= 0:
                self.new_speed = self.speed - speed_step
            else:
                self.new_speed = 0
            self.new_speed = max(self.new_speed, -100)
        elif char == "c":
            self.new_speed = 0
            self.new_turn = 0
        elif char == "q":
            self.new_speed = 0
            self.new_turn = 0
            self.is_running = False
        elif char == "m":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
        ### You can add you own modes here ###
        ## Example: ##
        # elif char == "r":
        #    self.mode = TaskPart.Enum
        #    print("[green]MODE: Enum")
        ###

        if self.speed != self.new_speed or self.turn != self.new_turn:
            self.speed = self.new_speed
            self.turn = self.new_turn
            print("speed:", self.speed, "turn:", self.turn)


if __name__ == '__main__':

    main = Main()

