import numpy as np
from rich import print
from timeit import default_timer as timer
import copy

def difference_angle(angle1, angle2):
    difference = (angle1 - angle2) % (2*np.pi)
    difference = np.where(difference > np.pi, difference - 2*np.pi, difference)
    difference = difference.item()  # convert 0-d array to float
    return difference

class EKFSLAM:
    def __init__(self,
                 WHEEL_RADIUS,
                 WIDTH,
                 MOTOR_STD,
                 DIST_STD,
                 ANGLE_STD,
                 init_state: np.ndarray = np.zeros(3),
                 init_covariance: np.ndarray = np.zeros((3,3))):
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        self.mu = init_state.copy()
        self.Sigma = init_covariance.copy()
        self.ids = np.full((1000,), -1)  # array instead of list
        self.ids_index = np.full((2000,), -1)  # lookup table for id indices
        self.num_ids = 0

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        self.error_l = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)
        self.error_r = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)

        self.num_times_seen_landmark = {}

    def predict(self, l, r):
        """
        EKF-SLAM: Prediction step (time update)

        l, r : traveled distance of left and right wheels (meters)
               typically from motor encoders.

        Idea:
        - Use l, r and robot geometry (WIDTH) to compute new pose (x, y, theta)
        - Compute Jacobians G and V to update covariance Sigma:
            Sigma' = G * Sigma * G^T + V * R * V^T
        (R is diag(error_l^2, error_r^2))
        """
        time1 = timer()

        # Get current robot pose
        x, y, theta, std = self.get_robot_pose()

        # Control: difference between left and right wheel distance
        alpha = (r - l) / self.WIDTH  # approximate rotation

        # Case distinction: curve vs straight
        if abs(r - l) >= np.radians(1) * self.WHEEL_RADIUS:  # curve if wheels differ significantly

            # --- Curved motion: Differential-drive kinematics
            R = l / alpha  # radius to Instantaneous Center of Curvature (ICC)

            # New pose after small rotation alpha
            x_new = x + (R + self.WIDTH/2) * (np.sin(theta + alpha) - np.sin(theta))
            y_new = y - (R + self.WIDTH/2) * (np.cos(theta + alpha) - np.cos(theta))
            theta_new = theta + alpha

            # Partial derivatives of motion function w.r.t wheel distances
            A = 0.5 * np.cos(theta + alpha)
            B = 0.5 * np.sin(theta + alpha)
            C = 0.5 * np.cos(theta + alpha)
            D = 0.5 * np.sin(theta + alpha)

            # Jacobian G of motion function w.r.t state
            G = np.identity(3)
            G[0, 2] = (R + self.WIDTH/2) * (np.cos(theta + alpha) - np.cos(theta))
            G[1, 2] = (R + self.WIDTH/2) * (np.sin(theta + alpha) - np.sin(theta))

            x, y, theta = x_new, y_new, theta_new

        else:
            # --- Straight motion
            x_new = x + l * np.cos(theta)
            y_new = y + l * np.sin(theta)
            theta_new = theta

            # Partial derivatives for straight motion
            A = np.cos(theta)
            B = np.sin(theta)
            C = np.cos(theta)
            D = np.sin(theta)

            G = np.identity(3)
            G[0, 2] = -l * np.sin(theta)
            G[1, 2] = l * np.cos(theta)

            x, y, theta = x_new, y_new, theta_new

        # Jacobian of motion w.r.t control inputs l and r
        V = np.array([
            [A, C],
            [B, D],
            [-1/self.WIDTH, 1/self.WIDTH]  # ∂theta/∂l, ∂theta/∂r
        ])

        # Extend G and V if landmarks exist
        N = self.num_ids
        if N > 0:
            G = np.block([
                [G, np.zeros((3, 2*N))],
                [np.zeros((2*N, 3)), np.identity(2*N)]
            ])
            V = np.append(V, np.zeros((2*N, 2)), axis=0)

        # Update robot state
        self.mu[:3] = x, y, theta

        # Covariance update
        diag = np.diag(np.array([self.error_l**2, self.error_r**2]))
        self.Sigma = G @ self.Sigma @ G.T + V @ diag @ V.T

        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM predict time: {elapsed_time}")

    def add_landmark(self, position: tuple, measurement: tuple, id: int):
        x, y = position
        x_var, y_var = 100.0, 100.0  # initial variance

        self.mu = np.append(self.mu, [x, y])
        self.Sigma = np.block([
            [self.Sigma, np.zeros((self.Sigma.shape[0], 2))],
            [np.zeros((2, self.Sigma.shape[1])), np.diag([x_var, y_var])]
        ])

        self.ids[self.num_ids] = id
        self.ids_index[id] = self.num_ids
        self.num_ids += 1
        self.num_times_seen_landmark[id] = 1

    def correction(self, landmark_position_measured: tuple, id: int, count=True):
        """
        EKF-SLAM: Correction step using a single landmark measurement
        """
        time1 = timer()
        r_meas, alpha_meas = landmark_position_measured

        N = self.num_ids
        i = self.ids_index[id]

        x_lm, y_lm = self.mu[3+2*i : 3+2*(i+1)]
        x_bot, y_bot, theta_bot = self.mu[:3].copy()

        dx = x_lm - x_bot
        dy = y_lm - y_bot
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        alpha = np.arctan2(dy, dx) - theta_bot

        H_small = np.zeros((2,5))
        H_small[0,:3] = [-dx / r, -dy / r, 0]
        H_small[1,:3] = [dy / r2, -dx / r2, -1]

        H_small[0,3:5] = [dx / r, dy / r]
        H_small[1,3:5] = [-dy / r2, dx / r2]

        H = np.zeros((2,3+2*N))
        H[0,:3] = [-dx / r, -dy / r, 0]
        H[1,:3] = [dy / r2, -dx / r2, -1]
        H[0,3+2*i:3+2*(i+1)] = [dx / r, dy / r]
        H[1,3+2*i:3+2*(i+1)] = [-dy / r2, dx / r2]

        sigma_small = np.zeros((5,5))
        sigma_small[:3,:3] = self.Sigma[:3,:3]
        sigma_small[3:5,3:5] = self.Sigma[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
        sigma_small[3:5,0:3] = self.Sigma[3+2*i:3+2*(i+1),0:3]
        sigma_small[0:3,3:5] = self.Sigma[0:3,3+2*i:3+2*(i+1)]

        # Measurement noise covariance Q
        Q = np.diag([self.DIST_STD**2, self.ANGLE_STD**2])

        # Innovation covariance
        Z = H_small @ sigma_small @ H_small.T + Q

        # Kalman gain
        K = self.Sigma @ (H.T @ np.linalg.inv(Z))

        diff_in_angle = difference_angle(alpha_meas, alpha)
        diff_in_r = r_meas - r
        err = np.array([diff_in_r, diff_in_angle])

        correction = K @ err

        self.mu += correction
        self.Sigma = (np.identity(3+2*N) - K @ H) @ self.Sigma

        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM correction time: {elapsed_time}")

        if count:
            self.num_times_seen_landmark[id] += 1

    def get_robot_pose(self):
        x, y, theta = self.mu[:3]
        sigma = self.Sigma[:2, :2]
        error = self.get_error_ellipse(sigma)
        return x, y, theta, error

    def get_landmark_poses(self, at_least_seen_num=2):
        positions = self.mu[3:].copy().reshape(-1, 2)
        errors = [self.get_error_ellipse(self.Sigma[3+2*i:3+2*i+2,3+2*i:3+2*i+2]) for i in range(self.num_ids)]

        if at_least_seen_num <= 1:
            return positions, np.array(errors), np.array(self.ids[:self.num_ids])
        else:
            mask = np.array([self.num_times_seen_landmark[id] >= at_least_seen_num for id in self.ids[:self.num_ids]])
            return positions[mask], np.array(errors)[mask], np.array(self.ids[:self.num_ids])[mask]

    def get_landmark_pose(self, id):
        i = self.ids_index[id]
        j = 3 + 2*i
        return self.mu[3+2*i:3+2*(i+1)], self.get_error_ellipse(self.Sigma[j:j+2,j:j+2])

    def get_error_ellipse(self, covariance):
        """
        Compute 2D error ellipse (standard deviations + orientation)
        """
        if covariance is None or not np.any(covariance) or not np.all(np.isfinite(covariance)):
            return 0.0, 0.0, 0.0

        cov = 0.5 * (np.array(covariance, dtype=float) + np.array(covariance, dtype=float).T)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = np.maximum(eigen_vals[idx], 0.0)
        vx0, vy0 = eigen_vecs[:, idx][:,0]
        angle = np.arctan2(vy0, vx0)
        sx, sy = np.sqrt(eigen_vals[0]), np.sqrt(eigen_vals[1]) if len(eigen_vals) > 1 else 0.0
        return sx, sy, angle

    def get_landmark_ids(self):
        return np.array(self.ids[:self.num_ids])

    def remove_blocks(self):
        for landmark_id in self.ids:
            if landmark_id > -1 and landmark_id >= 1000:
                self.remove_by_id(landmark_id)

    def remove_by_id(self, landmark_id):
        print("[red]removing landmark ", landmark_id)
        i = self.ids_index[landmark_id]
        self.ids = np.delete(self.ids, i)
        self.ids_index[landmark_id] = -1

        for an_idx, an_id in enumerate(self.ids):
            if an_id == -1:
                break
            self.ids_index[an_id] = an_idx

        del self.num_times_seen_landmark[landmark_id]
        self.num_ids -= 1

        self.mu = np.hstack((self.mu[:3+2*i], self.mu[3+2*(i+1):]))
        self.Sigma = np.block([
            [self.Sigma[:3+2*i, :3+2*i], self.Sigma[:3+2*i, 3+2*(i+1):]],
            [self.Sigma[3+2*(i+1):, :3+2*i], self.Sigma[3+2*(i+1):, 3+2*(i+1):]]
        ])
