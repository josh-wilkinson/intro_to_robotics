import numpy as np
from rich import print
from timeit import default_timer as timer


# ------------------------------------------------------------
# Utility: compute difference between two angles in [-pi, pi)
# ------------------------------------------------------------
def difference_angle(angle1, angle2):
    diff = (angle1 - angle2) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    return float(diff)


# ============================================================
# EKF SLAM CLASS
# ============================================================
class EKFSLAM:
    def __init__(self,
        WHEEL_RADIUS,
        WIDTH,
        MOTOR_STD,
        DIST_STD,
        ANGLE_STD,
        init_state=np.zeros(3),
        init_covariance=np.zeros((3, 3))
    ):
        # Robot parameters
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        # EKF state mean and covariance
        self.mu = init_state.copy()                  # [x, y, theta, x1, y1, ...]
        self.Sigma = init_covariance.copy()

        # Landmark bookkeeping
        self.ids = np.full(1000, -1)
        self.ids_index = np.full(2000, -1)
        self.num_ids = 0
        self.num_times_seen_landmark = {}

        # Measurement noise
        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        # Motion noise (wheel encoder noise)
        self.error_l = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)
        self.error_r = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)


    # ========================================================
    # PREDICTION STEP
    # ========================================================
    def predict(self, l, r):
        time1 = timer()

        x, y, theta = self.mu[:3]
        alpha = (r - l) / self.WIDTH

        # ----------------------------------------------------
        # Case 1: Turning motion
        # ----------------------------------------------------
        if abs(r - l) > np.radians(1) * self.WHEEL_RADIUS:
            theta_old = theta
            R = l / alpha

            # Motion model
            x += (R + self.WIDTH / 2) * (np.sin(theta_old + alpha) - np.sin(theta_old))
            y += (R + self.WIDTH / 2) * (-np.cos(theta_old + alpha) + np.cos(theta_old))
            theta += alpha

            # Normalize orientation
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # Jacobian wrt state
            G = np.array([
                [1, 0, (R + self.WIDTH / 2) * (np.cos(theta_old + alpha) - np.cos(theta_old))],
                [0, 1, (R + self.WIDTH / 2) * (np.sin(theta_old + alpha) - np.sin(theta_old))],
                [0, 0, 1]
            ])

            # Jacobian wrt control (from PDF)
            A = (self.WIDTH * r) / ((r - l)**2) * (np.sin(theta_old + alpha) - np.sin(theta_old)) \
                - (r + l) / (2 * (r - l)) * np.cos(theta_old + alpha)

            B = (self.WIDTH * r) / ((r - l)**2) * (-np.cos(theta_old + alpha) + np.cos(theta_old)) \
                - (r + l) / (2 * (r - l)) * np.sin(theta_old + alpha)

            C = -(self.WIDTH * l) / ((r - l)**2) * (np.sin(theta_old + alpha) - np.sin(theta_old)) \
                + (r + l) / (2 * (r - l)) * np.cos(theta_old + alpha)

            D = -(self.WIDTH * l) / ((r - l)**2) * (-np.cos(theta_old + alpha) + np.cos(theta_old)) \
                + (r + l) / (2 * (r - l)) * np.sin(theta_old + alpha)

        # ----------------------------------------------------
        # Case 2: Straight motion
        # ----------------------------------------------------
        else:
            theta_old = theta
            x += l * np.cos(theta_old)
            y += l * np.sin(theta_old)

            G = np.array([
                [1, 0, -l * np.sin(theta_old)],
                [0, 1,  l * np.cos(theta_old)],
                [0, 0, 1]
            ])

            A = 0.5 * (np.cos(theta_old) + (l / self.WIDTH) * np.sin(theta_old))
            B = 0.5 * (np.sin(theta_old) - (l / self.WIDTH) * np.cos(theta_old))
            C = 0.5 * (np.cos(theta_old) - (l / self.WIDTH) * np.sin(theta_old))
            D = 0.5 * (np.sin(theta_old) + (l / self.WIDTH) * np.cos(theta_old))

        # Control Jacobian
        V = np.array([[A, C], [B, D], [-1 / self.WIDTH, 1 / self.WIDTH]])

        # Expand Jacobians if landmarks exist
        N = self.num_ids
        if N > 0:
            G = np.block([[G, np.zeros((3, 2 * N))],
                          [np.zeros((2 * N, 3)), np.eye(2 * N)]])
            V = np.vstack([V, np.zeros((2 * N, 2))])

        # Update mean and covariance
        self.mu[:3] = x, y, theta
        Q = np.diag([self.error_l**2, self.error_r**2])
        self.Sigma = G @ self.Sigma @ G.T + V @ Q @ V.T

        # Sanity check
        assert np.all(np.isfinite(self.mu))
        assert np.all(np.isfinite(self.Sigma))

        if timer() - time1 > 0.1:
            print("[red]EKF predict slow")


    # ========================================================
    # ADD LANDMARK
    # ========================================================
    def add_landmark(self, position, id):
        x, y = position

        self.mu = np.append(self.mu, [x, y])
        self.Sigma = np.block([
            [self.Sigma, np.zeros((self.Sigma.shape[0], 2))],
            [np.zeros((2, self.Sigma.shape[1])), np.diag([100.0, 100.0])]
        ])

        self.ids[self.num_ids] = id
        self.ids_index[id] = self.num_ids
        self.num_ids += 1
        self.num_times_seen_landmark[id] = 1


    # ========================================================
    # CORRECTION STEP
    # ========================================================
    def correction(self, measurement, id):
        r_meas, alpha_meas = measurement
        i = self.ids_index[id]

        x, y, theta = self.mu[:3]
        xm, ym = self.mu[3 + 2*i : 3 + 2*i + 2]

        dx = xm - x
        dy = ym - y
        r2 = dx**2 + dy**2
        r = max(np.sqrt(r2), 1e-6)

        alpha = np.arctan2(dy, dx) - theta

        # Reduced Jacobian
        Hs = np.array([
            [-dx / r, -dy / r, 0,  dx / r,  dy / r],
            [ dy / r2, -dx / r2, -1, -dy / r2, dx / r2]
        ])

        # Reduced covariance
        Sigma_s = np.zeros((5, 5))
        Sigma_s[:3, :3] = self.Sigma[:3, :3]
        Sigma_s[3:, 3:] = self.Sigma[3 + 2*i : 3 + 2*i + 2,
                                     3 + 2*i : 3 + 2*i + 2]
        Sigma_s[:3, 3:] = self.Sigma[:3, 3 + 2*i : 3 + 2*i + 2]
        Sigma_s[3:, :3] = self.Sigma[3 + 2*i : 3 + 2*i + 2, :3]

        Q = np.diag([self.DIST_STD**2, self.ANGLE_STD**2])
        S_inv = np.linalg.inv(Hs @ Sigma_s @ Hs.T + Q)

        # Full H
        H = np.zeros((2, len(self.mu)))
        H[:, :3] = Hs[:, :3]
        H[:, 3 + 2*i : 3 + 2*i + 2] = Hs[:, 3:]

        # Kalman gain
        K = self.Sigma @ H.T @ S_inv

        # Innovation
        innovation = np.array([
            r_meas - r,
            difference_angle(alpha_meas, alpha)
        ])

        # Update
        self.mu += K @ innovation
        self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi
        self.Sigma = (np.eye(len(self.mu)) - K @ H) @ self.Sigma

        assert np.all(np.isfinite(self.mu))
        assert np.all(np.isfinite(self.Sigma))
