from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary


class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Store the config as-is
        self.cam_config = cam_config
        
        # Dictionary for ArUco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Marker size (should be in meters, adjust based on your markers)
        self.marker_size = 0.05  # 5cm markers
        
        # Helper function to get config values
        def get_config_value(key, default):
            if hasattr(cam_config, key):
                return getattr(cam_config, key)
            elif isinstance(cam_config, dict) and key in cam_config:
                return cam_config[key]
            else:
                return default
        
        # Camera to robot transform (from cam_config)
        self.offset_camera_robot = np.array([
            get_config_value('x_offset', 0.03),
            get_config_value('y_offset', 0.0),
            get_config_value('z_offset', 0.28)
        ])
        
        self.rotation_camera_robot = np.array([
            np.radians(get_config_value('x_angle', -140)),
            np.radians(get_config_value('y_angle', 0)),
            np.radians(get_config_value('z_angle', -90))
        ])
        
        # Create camera to robot transformation matrix
        self.T_camera_robot = self._create_transform_matrix(
            self.rotation_camera_robot, 
            self.offset_camera_robot
        )

    def calculate_distance_and_heading(current_pose, goal_position):
        """Calculate distance and heading to goal"""
        x, y, theta, _ = current_pose
        dx = goal_position[0] - x
        dy = goal_position[1] - y
        distance = np.sqrt(dx**2 + dy**2)
        desired_heading = np.arctan2(dy, dx)
        heading_error = difference_angle(desired_heading, theta)
        return distance, heading_error
    
    def _rotation_matrix(self, x_rotation, y_rotation, z_rotation):
        """Create rotation matrix from Euler angles (intrinsic ZYX order)"""
        alpha = z_rotation  # rotation around z
        beta = y_rotation   # rotation around y
        gamma = x_rotation  # rotation around x
        
        return np.array([
            [np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
            [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
            [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]
        ])
    
    def _create_transform_matrix(self, rotation_euler, translation):
        """Create 4x4 transformation matrix from Euler angles and translation"""
        R = self._rotation_matrix(rotation_euler[0], rotation_euler[1], rotation_euler[2])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T
    
    def _vectors_to_transform_matrix(self, rotation_vec, translation_vec):
        """Convert rotation vector (axis-angle) and translation to 4x4 transform matrix"""
        R, _ = cv2.Rodrigues(rotation_vec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation_vec.flatten()
        return T
    
    def _transform_matrix_to_vectors(self, transform_matrix):
        """Convert 4x4 transform matrix to rotation vector and translation"""
        R = transform_matrix[:3, :3]
        translation = transform_matrix[:3, 3]
        rotation_vec, _ = cv2.Rodrigues(R)
        return rotation_vec, translation
    
    def _create_robot_to_world_transform(self, robot_pose):
        """Create transformation matrix from robot frame to world frame"""
        # robot_pose from slam.get_robot_pose() returns (x, y, theta, error)
        x, y, theta = robot_pose[0], robot_pose[1], robot_pose[2]
        
        # For 2D robots, we only have rotation around Z axis (yaw)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([x, y, 0])  # z = 0 for ground robots
        return T

    def detections(self, img: np.ndarray, draw_img: np.ndarray, robot_pose: tuple, kind: str = "aruco") -> tuple:
        """
        Detect landmarks and return their positions in world coordinates.
        
        Args:
            img: Input image (undistorted)
            draw_img: Image to draw detections on
            robot_pose: (x, y, theta) of robot in world coordinates (theta in radians)
            kind: Type of detection ("aruco" for ArUco markers)
            
        Returns:
            ids: List of landmark IDs
            landmark_rs: List of distances from robot to landmarks
            landmark_alphas: List of angles from robot to landmarks (relative to robot heading)
            landmark_positions: List of (x, y) positions in world coordinates
        """
        ids, landmark_rs, landmark_alphas, landmark_positions = [], [], [], []
        
        if kind.lower() == "aruco":
            # Detect ArUco markers
            corners, detected_ids, rejected = self.aruco_detector.detectMarkers(img)
            
            if detected_ids is not None:
                # Create marker points in marker coordinate system
                marker_points = np.array([
                    [-self.marker_size / 2, self.marker_size / 2, 0],
                    [self.marker_size / 2, self.marker_size / 2, 0],
                    [self.marker_size / 2, -self.marker_size / 2, 0],
                    [-self.marker_size / 2, -self.marker_size / 2, 0]
                ])
                
                # Create robot to world transform
                T_robot_world = self._create_robot_to_world_transform(robot_pose)
                
                for i in range(len(detected_ids)):
                    marker_id = detected_ids[i][0]
                    
                    # Estimate pose of marker relative to camera
                    success, rvec, tvec = cv2.solvePnP(
                        objectPoints=marker_points,
                        imagePoints=corners[i],
                        cameraMatrix=self.camera_matrix,
                        distCoeffs=self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    if success:
                        # Convert to camera-to-marker transform
                        T_camera_marker = self._vectors_to_transform_matrix(rvec, tvec)
                        
                        # Transform to robot frame: T_robot_marker = T_camera_robot * T_camera_marker
                        T_robot_marker = self.T_camera_robot @ T_camera_marker
                        
                        # Transform to world frame: T_world_marker = T_robot_world * T_robot_marker
                        T_world_marker = T_robot_world @ T_robot_marker
                        
                        # Get translation vector in world frame
                        _, tvec_world = self._transform_matrix_to_vectors(T_world_marker)
                        
                        # Calculate distance and angle relative to robot
                        # Position of marker relative to robot (in robot frame)
                        _, tvec_robot = self._transform_matrix_to_vectors(T_robot_marker)
                        
                        # Distance from robot to marker
                        r = np.sqrt(tvec_robot[0]**2 + tvec_robot[1]**2)
                        
                        # Angle from robot heading to marker (alpha)
                        # Robot heading is along positive X axis
                        alpha = np.arctan2(tvec_robot[1], tvec_robot[0])
                        
                        # Store results
                        ids.append(marker_id)
                        landmark_rs.append(r)
                        landmark_alphas.append(alpha)
                        landmark_positions.append((tvec_world[0], tvec_world[1]))
                        
                        # Draw on image for visualization
                        cv2.aruco.drawDetectedMarkers(draw_img, corners, detected_ids)
                        
                        # Draw axes on marker
                        cv2.drawFrameAxes(
                            image=draw_img,
                            cameraMatrix=self.camera_matrix,
                            distCoeffs=self.dist_coeffs,
                            rvec=rvec,
                            tvec=tvec,
                            length=0.03,
                            thickness=2
                        )
                        
                        # Add text with marker info
                        text = f"ID: {marker_id} Pos: ({tvec_world[0]:.2f}, {tvec_world[1]:.2f})"
                        cv2.putText(draw_img, text, 
                                   (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        print(f"[green]Detected marker {marker_id}:[/green] "
                              f"World position: ({tvec_world[0]:.3f}, {tvec_world[1]:.3f}) m, "
                              f"Distance: {r:.3f} m, Angle: {np.degrees(alpha):.1f}°")
                    else:
                        print(f"[red]Failed to estimate pose for marker {marker_id}[/red]")
            else:
                print("[yellow]No ArUco markers detected[/yellow]")
        
        return ids, landmark_rs, landmark_alphas, landmark_positions