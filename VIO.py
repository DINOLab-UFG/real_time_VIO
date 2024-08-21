import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

class InertialOdometry:
    def __init__(self, P, Q):
        self.P = P
        self.Q = Q

    def imu_predict(self, state, imu_data, dt):
        p = state['position']
        v = state['velocity']
        q = state['orientation']

        acc = imu_data['acceleration']
        gyro = imu_data['gyroscope']

        g = np.array([0, 0, 9.81])
        acc_world = R.from_quat(q).apply(acc)
        v_ = v + (acc_world - g) * dt
        p = p + v * dt

        omega = R.from_rotvec(gyro * dt)
        q = (R.from_quat(q) * omega).as_quat()
        q /= np.linalg.norm(q)

        return {
            'position': p,
            'velocity': v_,
            'orientation': q
        }

    def compute_jacobian(self, state, imu_data, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def update_covariance(self, F):
        self.P = F @ self.P @ F.T + self.Q
        return self.P

    def predict(self, state, imu_data, dt):
        new_state = self.imu_predict(state, imu_data, dt)
        F = self.compute_jacobian(state, imu_data, dt)
        self.update_covariance(F)
        return new_state, self.P

class KalmanFilter:
    def __init__(self, state_dim, meas_dim):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self.x = np.zeros((state_dim, 1))
        self.P = np.eye(state_dim)
        self.F = np.eye(state_dim)
        self.H = np.zeros((meas_dim, state_dim))
        self.Q = np.eye(state_dim) * 0.01
        self.R = np.eye(meas_dim) * 0.1
        self.K = np.zeros((state_dim, meas_dim))

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + self.K @ y
        self.P = (np.eye(self.state_dim) - self.K @ self.H) @ self.P

class VisualOdometry:
    def __init__(self, data_dir, imu_odometry):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.kalman = KalmanFilter(state_dim=6, meas_dim=6)
        self.kalman.F = np.eye(6)
        self.kalman.H = np.eye(6)
        self.cur_pose = np.eye(4)

        # Inertial Odometry instance
        self.imu_odometry = imu_odometry

    def _load_calib(self, filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def _load_poses(self, filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    def _load_images(self, filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def get_matches(self, i):
        p0 = cv2.goodFeaturesToTrack(self.images[i - 1], mask=None, **self.feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.images[i - 1], self.images[i], p0, None, **self.lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        return good_old, good_new

    def _form_transf(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1, np.ndarray.flatten(t))
        T2 = self._form_transf(R2, np.ndarray.flatten(t))
        T3 = self._form_transf(R1, np.ndarray.flatten(-t))
        T4 = self._form_transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate((self.K, np.zeros((3, 1))), axis=1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max_idx = np.argmax(positives)
        if max_idx == 0:
            return R1, np.ndarray.flatten(t)
        elif max_idx == 1:
            return R2, np.ndarray.flatten(t)
        elif max_idx == 2:
            return R1, np.ndarray.flatten(-t)
        elif max_idx == 3:
            return R2, np.ndarray.flatten(-t)

    def update_kalman_filter(self, imu_state):
        x, y, z = imu_state['position']
        roll, pitch, yaw = imu_state['orientation']

        z = np.array([[x], [y], [z], [roll], [pitch], [yaw]])
        self.kalman.predict()
        self.kalman.update(z)

        updated_state = self.kalman.x

        self.cur_pose[0, 3], self.cur_pose[1, 3], self.cur_pose[2, 3] = updated_state[0, 0], updated_state[1, 0], updated_state[2, 0]
        self.cur_pose[:3, :3] = cv2.Rodrigues(np.array([updated_state[3, 0], updated_state[4, 0], updated_state[5, 0]]))[0]

    def run(self, imu_data_list, dt_list):
        traj = np.zeros((600, 600, 3), dtype=np.uint8)
        gt_path = []
        estimated_path = []

        state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': R.from_euler('xyz', [0, 0, 0]).as_quat()
        }

        for i in tqdm(range(1, len(self.images))):
            prev_img = self.images[i - 1]
            cur_img = self.images[i]
            prev_pts, cur_pts = self.get_matches(i)
            E, mask = cv2.findEssentialMat(cur_pts, prev_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, cur_pts, prev_pts, self.K)
            self.cur_pose = self.cur_pose @ self._form_transf(R, np.squeeze(t))

            state, _ = self.imu_odometry.predict(state, imu_data_list[i - 1], dt_list[i - 1])
            self.update_kalman_filter(state)

            x, y, z = self.cur_pose[0, 3], self.cur_pose[1, 3], self.cur_pose[2, 3]
            gt_x, gt_y = self.gt_poses[i][0, 3], self.gt_poses[i][2, 3]
            estimated_path.append((x, z))
            gt_path.append((gt_x, gt_y))

def main():
    data_dir = 'path'

    P = np.eye(6)
    Q = np.eye(6) * 0.01

    imu_odometry = InertialOdometry(P, Q)
    vo = VisualOdometry(data_dir, imu_odometry)

    imu_data_list = []  # Load your IMU data here
    dt_list = []  # Load your time intervals (dt) here
    vo.run(imu_data_list, dt_list)

if __name__ == '__main__':
    main()
