import numpy as np
from rich import print
from timeit import default_timer as timer
import copy
import pickle
import json

def difference_angle(angle1, angle2):
    difference = (angle1 - angle2) % (2*np.pi)
    difference = np.where(difference > np.pi, difference - 2*np.pi, difference)
    difference = difference.item() # we want to convert 0-d array to float

    return difference

class EKFSLAM:
    def __init__(self,
        WHEEL_RADIUS,
        WIDTH,

        MOTOR_STD,
        DIST_STD,
        ANGLE_STD,

        init_state: np.ndarray = np.zeros(3),
        init_covariance: np.ndarray = np.zeros((3,3))
    ):
        self.MOTOR_STD = MOTOR_STD
        self.DIST_STD = DIST_STD
        self.ANGLE_STD = ANGLE_STD
        self.WHEEL_RADIUS = WHEEL_RADIUS
        self.WIDTH = WIDTH

        self.mu = init_state.copy()
        self.Sigma = init_covariance.copy()
        self.ids = np.full((1000,), -1) # create an array instead of a list
        self.ids_index = np.full((2000,), -1) # lookup table to get index of id # assume, no ids larger than 2000
        self.num_ids = 0
        

        self.DIST_STD = DIST_STD
        self.ANGLE_STD = np.radians(ANGLE_STD)

        self.error_l = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)
        self.error_r = np.radians(MOTOR_STD) * WHEEL_RADIUS * np.sqrt(2)

        self.num_times_seen_landmark = {}

        return

    def predict(self, l, r):
        time1 = timer()

        x, y, theta, std = self.get_robot_pose()
         # l, r = u
        alpha = (r-l) / self.WIDTH

        if abs(r - l) >= np.radians(1) * self.WHEEL_RADIUS: # difference in left and right angle > 1°
            theta_old = theta
            R = l / alpha

            x += (R + self.WIDTH / 2) * (np.sin(theta_old + alpha) - np.sin(theta_old))
            y += (R + self.WIDTH / 2) * (-np.cos(theta_old + alpha) + np.cos(theta_old))
            theta += alpha

            G = np.array([
                [1, 0, (R + self.WIDTH / 2) * (np.cos(theta_old + alpha) - np.cos(theta_old))],
                [0, 1, (R + self.WIDTH / 2) * (np.sin(theta_old + alpha) - np.sin(theta_old))],
                [0, 0, 1]
            ])

            common = l / alpha + self.WIDTH / 2
            common_prime_l = 1 / alpha + l / (alpha**2 * self.WIDTH)
            common_prime_r = -l / (alpha**2 * self.WIDTH)

            A = common_prime_l * (np.sin(theta_old + alpha) - np.sin(theta_old)) + common * (-np.cos(theta_old + alpha) / self.WIDTH)
            B = common_prime_l * (-np.cos(theta_old + alpha) + np.cos(theta_old)) + common * (np.sin(theta_old + alpha) * (-1 / self.WIDTH))
            C = common_prime_r * (np.sin(theta_old + alpha) - np.sin(theta_old)) + common * (np.cos(theta_old + alpha) * (1 / self.WIDTH))
            D = common_prime_r * (-np.cos(theta_old + alpha) + np.cos(theta_old)) + common * (np.sin(theta_old + alpha) * (1 / self.WIDTH))
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
        V = np.array([[A, C], [B, D], [-1/self.WIDTH, 1/self.WIDTH]])

        N = self.num_ids
        if N > 0:
            G = np.block([[G, np.zeros((3,2*N))], [np.zeros((2*N,3)), np.identity(2*N)]])
            V = np.append(V, np.zeros((2*N,2)), axis=0)

        self.mu[:3] = x, y, theta

        diag = np.diag(np.array([np.power(self.error_l,2), np.power(self.error_r,2)]))
        self.Sigma = np.dot(np.dot(G, self.Sigma), G.T) + np.dot(np.dot(V, diag), V.T)

        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM predict time: {elapsed_time}")

    def add_landmark(self, position: tuple, measurement: tuple, id: int):
        x, y = position
        # add with variance of 100m
        x_var = float(100)
        y_var = float(100)

        self.mu = np.append(self.mu, [x,y])
        self.Sigma = np.block([
            [self.Sigma, np.zeros((self.Sigma.shape[0], 2))],
            [np.zeros((2, self.Sigma.shape[1])), np.diag(np.array([x_var, y_var]))]
        ])

        # add the id to the array
        self.ids[self.num_ids] = id
        self.ids_index[id] = self.num_ids
        self.num_ids += 1

        self.num_times_seen_landmark[id] = 1

    def correction(self, landmark_position_measured: tuple, id: int, count=True):
        time1 = timer()

        r_meas, alpha_meas = landmark_position_measured

        N = self.num_ids
        i = self.ids_index[id]

        x_lm, y_lm = self.mu[3+2*i : 3+2*(i+1)]
        # x_bot, y_bot, theta_bot, stdev_bot = self.get_robot_pose()
        x_bot,y_bot,theta_bot = self.mu[:3].copy()

        dx = x_lm - x_bot
        dy = y_lm - y_bot

        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        alpha = np.arctan2(dy,dx) - theta_bot

        H_small = np.zeros((2,5))
        H_small[0,:3] = np.array([-dx / r, -dy / r, 0])
        H_small[1,:3] = np.array([dy / r**2, -dx / r**2, -1])

        H_small[0,3 : 5] = np.array([dx / r, dy / r])
        H_small[1,3 : 5] = np.array([-dy / r**2, dx / r**2])

        H = np.zeros((2,3+2*N))
        H[0,:3] = np.array([-dx / r, -dy / r, 0])
        H[1,:3] = np.array([dy / r2, -dx / r2, -1])

        H[0,3+2*i : 3+2*(i+1)] = np.array([dx / r, dy / r])
        H[1,3+2*i : 3+2*(i+1)] = np.array([-dy / r2, dx / r2])

        sigma_small = np.zeros((5, 5))

        sigma_small[:3, :3] = self.Sigma[:3, :3]
        sigma_small[3:5, 3:5] = self.Sigma[3+2*i : 3+2*(i+1), 3+2*i : 3+2*(i+1)]

        # cross-variances
        sigma_small[3:5, 0:3] = self.Sigma[3+2*i : 3+2*(i+1), 0:3]
        sigma_small[0:3, 3:5] = self.Sigma[0:3, 3+2*i : 3+2*(i+1)]

        ### Your code here ###
        # should include Q, Z, K

        Q = np.diag(np.array([self.DIST_STD**2, self.ANGLE_STD**2]))
        Z = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(Z)

        ###

        diff_in_angle = difference_angle(alpha_meas, alpha)
        diff_in_r = r_meas - r

        err = np.array([diff_in_r, diff_in_angle])
        correction = K @ err

        self.mu += correction

        self.Sigma = (np.identity(3+2*N) - K @ H) @ self.Sigma


        elapsed_time = timer() - time1
        if elapsed_time > 0.1:
            print(f"[red]EKF_SLAM correction time: {elapsed_time}")

        # count how many times we have found it
        if count:
            self.num_times_seen_landmark[id] += 1

    def get_robot_pose(self, fast=False):
        x,y,theta = self.mu[:3]
        sigma = self.Sigma[:2, :2]
        if fast:
            return x, y, theta, [0,0,0]
        error = self.get_error_ellipse(sigma)

        return x, y, theta, error

    def get_landmark_poses(self, at_least_seen_num=2):
        positions = self.mu[3:].copy()
        positions = positions.reshape((len(positions) // 2, 2))

        errors = []
        for i in np.arange(self.num_ids):
            j = 3 + 2 * i
            sigma_i = self.Sigma[j:j+2, j:j+2]

            errors.append(self.get_error_ellipse(sigma_i))

        if at_least_seen_num == 0 or at_least_seen_num == 1:
            return positions, np.array(errors), np.array(self.ids[:self.num_ids])
        else:
            # only return landmarks that have been seen at least twice
            mask = np.zeros(self.num_ids, dtype=bool)
            for i, id in enumerate(self.ids):
                if id == -1:
                    break
                if self.num_times_seen_landmark[id] >= at_least_seen_num:
                    mask[i] = 1

            return positions[mask], np.array(errors)[mask], np.array(self.ids[:self.num_ids])[mask]

    def get_landmark_pose(self, id):
        i = self.ids_index[id]
        positions = self.mu[3:]

        j = 3 + 2 * i
        sigma_i = self.Sigma[j:j+2, j:j+2]

        return positions[i*2:i*2 + 2], self.get_error_ellipse(sigma_i)

    def get_error_ellipse(self, covariance):
        all_zeros = not np.any(covariance)
        if all_zeros:
            return 0, 0, 0
        else:
            eigen_vals, eigen_vecs = np.linalg.eig(covariance)

            # angle of first (largest) eigenvector
            angle = np.arctan2(eigen_vecs[1, 0], eigen_vecs[0, 0])

        return np.sqrt(eigen_vals[0]), np.sqrt(eigen_vals[1]), angle

    def get_landmark_ids(self):
        return np.array(self.ids[:self.num_ids])

    def remove_blocks(self):
        for landmark_id in self.ids:
            if landmark_id > -1 and landmark_id >= 1000:
                self.remove_by_id(landmark_id)
                
    def remove_by_id(self, landmark_id):

        print("[red]removing landmark ", landmark_id)

        # get index
        i = self.ids_index[landmark_id]

        self.ids = np.delete(self.ids, i) # same as pop

        # recreate index array
        self.ids_index[landmark_id] = -1
        for an_idx, an_id in enumerate(self.ids):
            if an_id == -1:
                break

            self.ids_index[an_id] = an_idx

        del self.num_times_seen_landmark[landmark_id]

        self.num_ids -= 1

        # remove the block from mu
        self.mu = np.hstack((self.mu[:3+2*i], self.mu[3+2*(i+1):]))

        # we remove the row and column corresponding to the 2x2 Sigma block
        self.Sigma = np.block([
            [self.Sigma[:3+2*i, :3+2*i], self.Sigma[:3+2*i, 3+2*(i+1):]],
            [self.Sigma[3+2*(i+1):, :3+2*i], self.Sigma[3+2*(i+1):, 3+2*(i+1):]]
        ])

    def save_state(self, filename="EKFSLAM.json"):
        "Save the current state of the EKF-SLAM map, etc. to file in a somewhat human readale format."
        state = {
            'mu': self.mu.tolist(),
            'Sigma': self.Sigma.tolist(),
            'ids': self.ids.tolist(),
            'ids_index': self.ids_index.tolist(),
            'num_ids': self.num_ids,
            'num_times_seen_landmark': self.num_times_seen_landmark,
            'WHEEL_RADIUS': self.WHEEL_RADIUS,
            'WIDTH': self.WIDTH,
            'MOTOR_STD': self.MOTOR_STD,
            'DIST_STD': self.DIST_STD,
            'ANGLE_STD': self.ANGLE_STD,
        }
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
        else:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=4)

    @classmethod
    def load_state(cls, filename="EKFSLAM.json"):
        "Load the EKF-SLAM map, etc. from a previously saved state."
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                    state = pickle.load(f)
        else:
            with open(filename, 'r') as f:
                state = json.load(f)
        ekf_slam = cls(
            WHEEL_RADIUS=state['WHEEL_RADIUS'],
            WIDTH=state['WIDTH'],
            MOTOR_STD=state['MOTOR_STD'],
            DIST_STD=state['DIST_STD'],
            ANGLE_STD=state['ANGLE_STD'],
            init_state=np.array(state['mu']),
            init_covariance=np.array(state['Sigma'])
        )
        ekf_slam.ids = np.array(state['ids'])
        ekf_slam.ids_index = np.array(state['ids_index'])
        ekf_slam.num_ids = state['num_ids']
        ekf_slam.num_times_seen_landmark = {
            int(k): v for k, v in state['num_times_seen_landmark'].items()
        }

        return ekf_slam
