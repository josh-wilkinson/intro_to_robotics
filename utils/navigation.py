"""
Purpose: Navigation system for driving to real-world coordinates
"""

import numpy as np
import time
from typing import Tuple, Optional
from collections import deque

from utils.utils import difference_angle


class PIDController:
    """Enhanced PID controller for navigation"""
    # (Use your existing PIDController class or import from notebook)
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0
        self.previous_error = 0
        self.previous_time = None
        
    def update(self, error, dt):
        # PID implementation
        pass


class NavigationController:
    """
    Main navigation controller for driving to world coordinates
    """
    
    def __init__(self, config):
        self.config = config
        
        # PID controllers
        self.distance_pid = PIDController(
            kp=config.navigation.distance_kp,
            ki=config.navigation.distance_ki,
            kd=config.navigation.distance_kd,
            output_limits=(0, 100)
        )
        
        self.heading_pid = PIDController(
            kp=config.navigation.heading_kp,
            ki=config.navigation.heading_ki,
            kd=config.navigation.heading_kd,
            output_limits=(-100, 100)
        )
        
        # Navigation state
        self.current_goal = None
        self.goal_reached = False
        self.path_history = []
        
        # Parameters
        self.goal_threshold = getattr(config.navigation, 'goal_threshold', 0.05)
        self.max_speed = getattr(config.navigation, 'max_speed', 50)
        self.min_speed = getattr(config.navigation, 'min_speed', 10)
        self.arrival_distance = getattr(config.navigation, 'arrival_distance', 0.02)
        
    def set_goal(self, goal_position: Tuple[float, float]):
        """Set a new navigation goal"""
        self.current_goal = np.array(goal_position)
        self.goal_reached = False
        self.distance_pid.reset()
        self.heading_pid.reset()
        print(f"[Navigation] New goal set: {goal_position}")
        
    def compute_controls(self, robot_pose, dt=0.1) -> Tuple[float, float]:
        """
        Compute speed and turn commands based on current pose
        
        Args:
            robot_pose: (x, y, theta, _) from slam.get_robot_pose()
            dt: Time step in seconds
            
        Returns:
            speed, turn: Control signals for robot movement
        """
        if self.current_goal is None or self.goal_reached:
            return 0, 0
            
        x, y, theta, _ = robot_pose
        
        # Calculate distance to goal
        dx = self.current_goal[0] - x
        dy = self.current_goal[1] - y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Check if goal is reached
        if distance < self.arrival_distance:
            self.goal_reached = True
            print(f"[Navigation] Goal reached! Final distance: {distance:.3f}m")
            return 0, 0
        
        # Calculate desired heading and error
        desired_heading = np.arctan2(dy, dx)
        heading_error = difference_angle(desired_heading, theta)
        
        # Update distance PID (speed control)
        # Scale speed based on distance (slower when closer)
        distance_error = distance
        speed_raw = self.distance_pid.update(distance_error, dt)
        
        # Apply speed limits and distance-based scaling
        speed = np.clip(speed_raw, self.min_speed, self.max_speed)
        
        # Reduce speed when close to goal
        if distance < 0.2:  # Slow down zone
            speed = max(self.min_speed, speed * (distance / 0.2))
        
        # Update heading PID (turn control)
        turn = self.heading_pid.update(heading_error, dt)
        
        # Record path for debugging
        self.path_history.append({
            'timestamp': time.time(),
            'position': (x, y),
            'heading': theta,
            'distance_to_goal': distance,
            'heading_error': heading_error,
            'speed': speed,
            'turn': turn
        })
        
        return speed, turn
    
    def get_navigation_status(self):
        """Get current navigation status"""
        if self.current_goal is None:
            return "No goal set"
        elif self.goal_reached:
            return f"Goal reached at {self.current_goal}"
        else:
            if len(self.path_history) > 0:
                last_pos = self.path_history[-1]['position']
                distance = np.sqrt(
                    (last_pos[0] - self.current_goal[0])**2 + 
                    (last_pos[1] - self.current_goal[1])**2
                )
                return f"Navigating to {self.current_goal}, distance: {distance:.2f}m"
            return f"Navigating to {self.current_goal}"
    
    def get_path_history(self):
        """Get recorded path history"""
        return self.path_history
    
    def clear_path_history(self):
        """Clear path history"""
        self.path_history = []
        
    def reset(self):
        """Reset navigation controller"""
        self.current_goal = None
        self.goal_reached = False
        self.distance_pid.reset()
        self.heading_pid.reset()
        self.clear_path_history()


class WaypointNavigator:
    """
    Advanced navigator for following multiple waypoints
    """
    
    def __init__(self, config):
        self.nav_controller = NavigationController(config)
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_threshold = getattr(config.navigation, 'waypoint_threshold', 0.1)
        
    def set_waypoints(self, waypoints):
        """Set a list of waypoints to follow"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        if waypoints:
            self.nav_controller.set_goal(waypoints[0])
            
    def update(self, robot_pose, dt=0.1):
        """
        Update navigation for waypoint following
        
        Returns:
            speed, turn, waypoint_complete, all_complete
        """
        if not self.waypoints:
            return 0, 0, False, True
            
        # Check if current waypoint is reached
        x, y, theta, _ = robot_pose
        current_goal = self.waypoints[self.current_waypoint_index]
        distance = np.sqrt((x - current_goal[0])**2 + (y - current_goal[1])**2)
        
        if distance < self.waypoint_threshold:
            print(f"[WaypointNavigator] Waypoint {self.current_waypoint_index} reached")
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index >= len(self.waypoints):
                print("[WaypointNavigator] All waypoints completed")
                return 0, 0, True, True
            else:
                # Move to next waypoint
                next_goal = self.waypoints[self.current_waypoint_index]
                self.nav_controller.set_goal(next_goal)
                return 0, 0, True, False
        
        # Get controls for current waypoint
        speed, turn = self.nav_controller.compute_controls(robot_pose, dt)
        return speed, turn, False, False
    
    def get_current_waypoint(self):
        """Get current target waypoint"""
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    def get_progress(self):
        """Get navigation progress percentage"""
        if not self.waypoints:
            return 100
        return (self.current_waypoint_index / len(self.waypoints)) * 100