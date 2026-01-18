from types import SimpleNamespace
import numpy as np


class DummyColour():
    def __init__(self) -> None:
        self.intensity = 0.0
        self.ambient = 0.0



class DummyVehicle():
    def __init__(self, x=0, y=0, theta=0, wheel_radius=0, robot_width=0, dt=0, v_max=0, w_max=0, move_max=100, turn_max=200) -> None:
        self.motor_pos = SimpleNamespace()
        self.motor_pos.left = 0
        self.motor_pos.right = 0
        
        
        self.x = x
        self.y = y
        self.theta = theta

        # robot dimensions
        self.wheel_radius = wheel_radius
        self.robot_width = robot_width
        
        self.dt = dt
        self.v_max = v_max
        self.w_max = w_max # in rad/s

        self.move_max = move_max
        self.turn_max = turn_max

        # the current rotation of the wheels
        self.l = 0
        self.r = 0

    def move(self, speed, turn):
        v = self.v_max * speed / 100
        w = self.w_max * turn / 100

        if w == 0:
            vl = v
            vr = v
        else:
            vl = w*(v/w + self.robot_width/2)
            vr = w*(v/w - self.robot_width/2)

        self.l = self.l + self.wheel_rotation_from_distance(vl * self.dt)
        self.r = self.r + self.wheel_rotation_from_distance(vr * self.dt)

        self.forward_kinematics(self.wheel_rotation_from_distance(vl * self.dt), self.wheel_rotation_from_distance(vr * self.dt))

    def forward_kinematics(self, l, r):
        # l, r = u
        alpha = (r-l) / self.robot_width


        if abs(r - l) >= np.radians(1) * self.wheel_radius: # difference in left and right angle > 1°
            R = l / alpha
            self.x += (R + 0.5*self.robot_width) * (np.sin(self.theta + alpha) - np.sin(self.theta))
            self.y += (R + 0.5*self.robot_width) * (-np.cos(self.theta + alpha) + np.cos(self.theta))
            self.theta = (self.theta + alpha) % (2*np.pi)
        else:
            sint = np.sin(self.theta)
            cost = np.cos(self.theta)

            l = np.mean(np.array([l,r]))
            self.x += l * cost
            self.y += l * sint
    
    def get_robot_v(self, move):
        """
        @brief: Returns the current speed of the robot as a fraction of v_max, based on the move parameter. All changes are assumed to be instantaneous.
        @param move: Parameter that determines the movement speed. Limited to interval [-self.move_max, self.move_max]
        @return: The current speed
        """
        return np.clip(move, -1*self.move_max, self.move_max) / 100 * self.v_max
    

    def wheel_rotation_from_distance(self, d):
        """
        @brief: Calculates the rotation of a robot wheel from the distance that wheel has traveled and the wheel diameter
        @param d: Distance traveled
        @return: Angle by which the wheel has turned while traveling distance d
        """
        wheel_circumference = 2*self.wheel_radius*np.pi
        revolutions = d / wheel_circumference

        return 2*np.pi * revolutions

    def set_position(self, x, y, theta):
        """
        @brief: sets the robot position and heading to given values
        """
        self.x = x
        self.y = y
        self.theta = theta

    def stop(self, brake):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def move_robot(speed, turn, robot, slam):
    old_l = robot.l
    old_r = robot.r
    robot.move(speed, turn)
    l = robot.l
    r = robot.r

    movements = l - old_l, r - old_r
    if movements[0] != 0.0 or movements[1] != 0:
        slam.predict(*movements)
    robot_x, robot_y, robot_theta, robot_stdev = slam.get_robot_pose()

    robot.set_position(robot_x, robot_y, robot_theta)