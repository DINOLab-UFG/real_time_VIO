import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        # Detect keypoints in the i-1'th image
        p0 = cv2.goodFeaturesToTrack(self.images[i - 1], mask=None, **self.feature_params)

        # Calculate optical flow (track keypoints) in the i'th image
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.images[i - 1], self.images[i], p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        return good_old, good_new

    def get_pose(self, q1, q2):
        Essential, _ = cv2.findEssentialMat(q1, q2, self.K)
        R, t = self.decomp_essential_mat(Essential, q1, q2)
        return self._form_transf(R,t)

    def decomp_essential_mat(self, E, q1, q2):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]

        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        max = np.argmax(positives)
        if (max == 2):
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            return R2, np.ndarray.flatten(t)

def main():
    data_dir = 'path'
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\nCurrent pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))


if __name__ == "__main__":
    main()