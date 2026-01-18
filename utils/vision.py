from __future__ import annotations
import cv2
import numpy as np
from rich import print
from utils.opencv_utils import putBText
from scipy.spatial.transform import Rotation
from scipy import optimize
from enum import Enum
from utils.utils import boundary
import math
import math

class Vision:
    def __init__(self, camera_matrix, dist_coeffs, cam_config) -> None:

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.cam_config = cam_config

    def detections(self, img, draw_img, robot_pose, kind="aruco"):
        """
        Detect ArUco markers in the input camera frame and convert them into:
        - ids                : list[int]
        - landmark_rs        : list[float]         (distance robot -> landmark)
        - landmark_alphas    : list[float]         (bearing in robot coordinates)
        - landmark_positions : list[np.array([x,y])] (2D world coordinates)

        NOTE:
        The EKF-SLAM implementation used in this course works **strictly in 2D**.
        Therefore all landmark positions returned to SLAM MUST be 2D (x,y).

        draw_img is modified in place (markers and axes drawn on top).
        """

        # =========================================================
        # 1) Read camera configuration parameters
        # =========================================================
        cfg = self.cam_config

        def cfg_get(name, default=None):
            """Helper function: supports both dict-style and attribute-style access."""
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        marker_size = cfg_get("marker_size", 0.05)
        aruco_dict_id = cfg_get("aruco_dict", cv2.aruco.DICT_ARUCO_ORIGINAL)
        cam_pose_in_base = cfg_get("cam_pose_in_base", None)
        T_base_cam = cfg_get("T_base_cam", None)
        # Kamera sitzt nicht im Zentrum des Roboters, sondern:
        #   - x_offset: nach vorne (in Fahrtrichtung)
        #   - z_offset: Höhe über dem Boden
        #   - y_angle: Neigung der Kamera (Pitch) um die Querachse
        x_offset = cfg_get("x_offset", 0.0)
        z_offset = cfg_get("z_offset", 0.0)
        y_angle  = cfg_get("y_angle", 0.0)
        # =========================================================
        # 2) ArUco detection (OpenCV >= 4.7 API)
        # =========================================================
        aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        corners, ids, rejected = detector.detectMarkers(img)

        if ids is None or len(ids) == 0:
            return [], [], [], []

        # =========================================================
        # 3) Build camera pose in the world frame
        # something is wrong here, maybe calculation
        # =========================================================
        rx, ry, rtheta = robot_pose[:3]

        def T_from_xytheta(x, y, theta):
            """Create a 4x4 homogeneous transform from a 2D robot pose."""
            T = np.eye(4)
            c, s = np.cos(theta), np.sin(theta)
            T[:3, :3] = [[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]]
            T[:3, 3] = [x, y, 0]
            return T

        def T_from_cam_config(d):
            """Convert a dictionary (x,y,z, roll,pitch,yaw) into a 4x4 transform."""
            tx = d.get("x", 0)
            ty = d.get("y", 0)
            tz = d.get("z", 0)
            roll  = d.get("roll", 0)
            pitch = d.get("pitch", 0)
            yaw   = d.get("yaw", 0)

            R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            return T
        

        def T_from_offsets(x_off, z_off, y_ang):
            """
            Construct T_base_cam (4x4) from the simple course parameters.

            Coordinate conventions:
            - Robot base frame:
                x: forward, y: left, z: up
            - Camera frame (OpenCV):
                x: right,  y: down, z: forward

            We first align camera axes to the robot base frame,
            then apply an additional pitch rotation y_ang,
            and finally translate by (x_off, 0, z_off).
            """
            # Align camera axes to robot base:
            #   cam z (forward)   -> base x (forward)
            #   cam x (right)     -> base -y (right is negative left)
            #   cam y (down)      -> base -z (down is negative up)
            R_align = np.array([
                [0.0,  0.0,  1.0],
                [-1.0, 0.0,  0.0],
                [0.0, -1.0,  0.0]
            ])

            # Additional pitch (tilt) around base y-axis
            R_pitch = Rotation.from_euler("y", y_ang).as_matrix()

            R_base_cam = R_align @ R_pitch

            T = np.eye(4, dtype=float)
            T[:3, :3] = R_base_cam
            T[:3,  3] = np.array([x_off, 0.0, z_off])  # camera position in base frame
            return T



        # Build T_world_cam from robot pose and camera extrinsics
        T_world_base = T_from_xytheta(rx, ry, rtheta)
        # Determine T_base_cam
        if T_base_cam is None:
            if cam_pose_in_base is not None:
                T_base_cam = T_from_cam_config(cam_pose_in_base)
            else:
                # Fallback: use the simple course parameters (x_offset, z_offset, y_angle)
                T_base_cam = T_from_offsets(x_offset, z_offset, y_angle)   

        T_world_cam = T_world_base @ T_base_cam
        R_wc = T_world_cam[:3, :3]
        t_wc = T_world_cam[:3, 3]

        # =========================================================
        # 4) Estimate marker poses in camera coordinates
        # =========================================================
        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        #     corners,
        #     marker_size,
        #     self.camera_matrix,
        #     self.dist_coeffs
        # )
        # =========================================================
        # Replace estimatePoseSingleMarkers with solvePnP
        #         # =========================================================

        rvecs = []
        tvecs = []

        # 3D coordinates of the marker corners in marker coordinate frame
        # Marker lies in the XY-plane, Z = 0
        marker_half = marker_size / 2.0
        objp = np.array([
            [-marker_half,  marker_half, 0],
            [ marker_half,  marker_half, 0],
            [ marker_half, -marker_half, 0],
            [-marker_half, -marker_half, 0]
        ], dtype=np.float32)
        # try to use IPPE_SQUARE if available (better for square planar markers)
        pnp_flag = cv2.SOLVEPNP_ITERATIVE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE


        for c in corners:
            # Ensure correct shape: (4,2) float32
            c2d = c.reshape(-1, 2).astype(np.float32)

            # SolvePnP returns rvec and tvec
            ok, rvec, tvec = cv2.solvePnP(
                objp,
                c2d,
                self.camera_matrix,
                self.dist_coeffs,
                #flags=cv2.SOLVEPNP_ITERATIVE
                flags=pnp_flag
            )

            if not ok:
                continue

            rvecs.append(rvec.reshape(3))
            tvecs.append(tvec.reshape(3))

        # convert lists to arrays to mimic original API
        rvecs = np.array(rvecs)
        tvecs = np.array(tvecs)

        # =========================================================
        # 5) Output lists returned to SLAM
        # =========================================================
        ids_out = []
        landmark_rs = []
        landmark_alphas = []
        landmark_positions = []

        # =========================================================
        # 6) Process each detected marker with sanity checks
        # =========================================================
        for i, id_arr in enumerate(ids.flatten()):

            rvec = rvecs[i].reshape(3)
            tvec = tvecs[i].reshape(3)  # 3D position in camera coordinates

            # Skip obviously invalid camera poses
            if not np.isfinite(tvec).all():
                continue


            # -------------------------------------------------
            # 1) Camera frame (OpenCV) → simple 2D robot frame
            # -------------------------------------------------
            # OpenCV ArUco convention:
            #   x_cam: right
            #   y_cam: down
            #   z_cam: forward
            #
            # Robot 2D frame (what SLAM uses):
            #   x_robot: forward
            #   y_robot: left
            # Mapping:
            #   x_robot =  z_cam          (forward)
            #   y_robot = -x_cam          (right -> negative left)
            # -------------------------------------------------
            x_cam, y_cam, z_cam = float(tvec[0]), float(tvec[1]), float(tvec[2])

            x_robot = z_cam
            y_robot = -x_cam

            # -------------------------------------------------
            # 2) Polar measurement (r, alpha) in robot frame
            # -------------------------------------------------
            r = float(np.hypot(x_robot, y_robot))
            if not np.isfinite(r) or r < 0.05 or r > 5.0:
                # ignore clearly wrong measurements
                continue

            alpha = float(np.arctan2(y_robot, x_robot))  # bearing in robot frame

            # Skip if angle is not finite (just in case)
            if not np.isfinite(alpha):
                continue
            # robot pose in world: (rx, ry, rtheta)
            world_angle = rtheta + alpha

            xw = rx + r * np.cos(world_angle)
            yw = ry + r * np.sin(world_angle)

            if not np.isfinite(xw) or not np.isfinite(yw):
                continue
            # Store results
            ids_out.append(int(id_arr))
            landmark_rs.append(r)
            landmark_alphas.append(alpha)
            landmark_positions.append(np.array([xw, yw], dtype=float))  # 2D only

            # Visualization
            try:
                cv2.aruco.drawDetectedMarkers(draw_img, [corners[i]])
                cv2.aruco.drawAxis(draw_img,
                                self.camera_matrix,
                                self.dist_coeffs,
                                rvec,
                                tvec,
                                marker_size * 0.5)
                txt = f"ID:{int(id_arr)}  r={r:.2f}m  a={np.degrees(alpha):.0f}°"
                putBText(draw_img,
                        txt,
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10)))
            except Exception:
                # Never let visualization kill the SLAM loop
                pass

        return ids_out, landmark_rs, landmark_alphas, landmark_positions
