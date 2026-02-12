import cv2
import time
import numpy as np
import sys
import jsonpickle
import pickle
from message import Message
from timeit import default_timer as timer
from collections import deque

from utils.robot_controller import RobotController
from utils.border_exploring import ExplorationController

from publisher import Publisher
from utils.keypress_listener import KeypressListener
from rich import print
from utils.utils import load_config
from utils.pathplanning import ExpandableGrid, read_target

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
    Searching_for_Landmarks = 1
    Mapping_outer_Border = 2
    Localizing = 3
    Navigating_to_Waypoints = 4
    Picking_up_Objects = 5
    Delivering_Objects = 6
    
    Saving_State = 99
    Reload_previous_State = 100

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

        self.explorer = ExplorationController()

        # set default values
        self.DT = self.config.robot.delta_t # delta time in seconds

        self.speed = 70
        self.turn = 10
        self.new_speed = 0
        self.new_turn = 0
        self.grid_config = {
            'resolution_m': 0.03,
            'width_m': 2.0,
            'height_m': 2.0,
            'offset_x': -1.0,
            'offset_y': -1.0,
            'expand_margin_m': 0.5,
        }

        self.manualMode = False
        self.is_running = True
        self.lpressed = False

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
        lower_blue = np.array([66, 230, 245])
        upper_blue = np.array([135, 66, 245])
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

    def _is_within_tolerance_of_path(self, position, path, tolerance):
        """
        Check if the given position is within a certain tolerance of the line between the waypoints in the path.
        position: (x, y)
        """
        pos = np.array(position)
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])

            line_vec = end - start
            point_vec = pos - start

            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue

            line_unitvec = line_vec / line_len
            projection_length = np.dot(point_vec, line_unitvec)
            projection_length = np.clip(projection_length, 0, line_len)

            closest_point = start + projection_length * line_unitvec
            distance = np.linalg.norm(pos - closest_point)

            if distance > tolerance:
                return False

        return True

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
            replan_needed = False

            while self.is_running:
                time0 = timer()
                self.run(count, time0)
                if count%10==0 or replan_needed:
                    target = read_target()
                    if target != None:
                        positions, _errors, ids = self.robot.slam.get_landmark_poses(at_least_seen_num=1)
                        grid = ExpandableGrid(self.grid_config)
                        for (x, y), mid in zip(positions, ids):
                            grid.add_marker(float(x), float(y), int(mid))

                        path_world = grid.plan_path_astar(self.robot.slam.get_robot_pose()[:2], [float(target[0]), float(target[1])])
                        print(f"[blue]Planned path to target {target} with {len(path_world)} waypoints.")
                        self.waypoints = path_world
                        replan_needed = False
                    if not self._is_within_tolerance_of_path(self.robot.slam.get_robot_pose()[:2], self.waypoints, tolerance=0.1):
                        print("[red] Outside of allowed trajectory corridor! Replan needed.")
                        self.waypoints = []
                        replan_needed = True


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
            raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
            #return

        draw_img = raw_img.copy()
        data = None
        # Once you have implemented EKF slam, you can use the data to create messages for the viewer
        data = self.robot.run_ekf_slam(raw_img, draw_img)

        self.parse_keypress()

        if self.mode == TaskPart.Manual:
            self.robot.move(self.new_speed, self.new_turn)

        if self.mode == TaskPart.Mapping_outer_Border:
            (self.new_speed, self.new_turn)=self.explorer.compute_controls(data, self.DT)
            if self.new_speed == 0:
                self.new_turn = 0
                self.mode = TaskPart.Manual
                print("[green]MODE: Manual")
            self.robot.move(self.new_speed, self.new_turn)


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
            if self.robot.recorder.is_recording:
                self.robot.recorder.save_step(raw_img, speed, turn,
                                          x=data.robot_position[0],
                                          y=data.robot_position[1],
                                          theta=data.robot_theta,
                                          l=self.robot.old_l,
                                          r=self.robot.old_r)

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

                robot_position = [0, 0, 0],
                robot_theta = 0,
                robot_stdev = [0, 0, 0],

                text = f"cam fps: {cam_fps}"
            )

        msg_str = jsonpickle.encode(msg)

        # send message to the viewer
        #print(msg_str)
        self.publisher.publish_img(msg_str, draw_img)


    def save_state(self, data):
        ### Your code here ###
        with open("SLAM.pickle", 'wb') as pickle_file:
            pickle.dump(data, pickle_file)


    def load_and_localize(self):
        ### Your code here ###
        with open("SLAM.pickle", 'rb') as f:
            data = pickle.load(f)
        return data


    def parse_keypress(self):
        char = self.keypress_listener.get_keypress()

        turn_step = 3
        speed_step = 5

        if char == "a":
            if self.turn >= 0:
                self.new_turn = self.turn - turn_step
            else:
                self.new_turn = 0
            self.new_turn = min(self.new_turn, 200)
        elif char == "d":
            if self.turn <= 0:
                self.new_turn = self.turn + turn_step
            else:
                self.new_turn = 0
            self.new_turn = max(self.new_turn, -200)
        elif char == "w":
            self.new_speed = self.speed + speed_step
            self.new_speed = min(self.new_speed, 100)
        elif char == "s":
            self.new_speed = self.speed - speed_step
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
        elif char == "b":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Mapping_outer_Border
            print("[green]MODE: Mapping_outer_Border")
        elif char == "l" or self.lpressed==False:
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Reload_previous_State
            print("[green]MODE: Reload_previous_State")
            data = self.load_and_localize()
            self.robot.slam.position_is_initialized = False
            self.robot.slam.set_state(data)
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
            self.lpressed=True
        elif char == "p":
            self.new_speed = 0
            self.new_turn = 0
            self.mode = TaskPart.Saving_State
            print("[green]MODE: Saving_State")
            data = self.robot.slam.get_state()
            self.save_state(data)
            self.mode = TaskPart.Manual
            print("[green]MODE: Manual")
        elif char == "r":
            if not self.robot.recorder.is_recording:
                self.robot.recorder.start_recording()
                print("[green]Recording started")
            else:
                self.robot.recorder.stop_recording()
                print("[green]Recording stopped and saved")

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

