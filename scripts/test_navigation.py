#!/usr/bin/env python3
"""
Test script for navigation system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
from types import SimpleNamespace
from utils.camera import Camera
from utils.vision import Vision
from utils.robot_controller import RobotController


def main():
    # Load config
    with open("../config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    
    print("Testing navigation system...")
    
    # Create robot controller
    with RobotController(config) as robot:
        # Test 1: Drive to a single point
        print("\n=== Test 1: Single point navigation ===")
        success = robot.drive_to_point((1.0, 0.5))
        print(f"Success: {success}")
        
        # Test 2: Follow multiple waypoints
        print("\n=== Test 2: Waypoint navigation ===")
        waypoints = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0)
        ]
        success = robot.follow_waypoints(waypoints)
        print(f"Success: {success}")
        
        print("\n=== Navigation test complete ===")


if __name__ == "__main__":
    main()