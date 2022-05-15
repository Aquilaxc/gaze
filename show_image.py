import cv2
import os
from utils_function import get_box
from scipy.io import loadmat
import numpy as np
from gazehub.gazeconversion import Gaze3DTo2D


def adjust_box(height, width, box, resize):
    x1, y1, x2, y2 = box * resize
    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


class ShowImage:
    def __init__(self, data_root, resize=0.4, resolution=(1440, 900)):
        self.imgpath = []
        self.labels = []
        self.monitor_pose = []
        self.resize = resize
        self.res = resolution
        for ii in range(15):
            person = ('p' + str(ii).rjust(2, '0'))
            label_path = os.path.join(data_root, person, person + ".txt")
            calibration = loadmat(os.path.join(data_root, person, "Calibration", "screenSize.mat"))
            monitorPose = loadmat(os.path.join(data_root, person, "Calibration", "monitorPose.mat"))
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip().split()
                    self.imgpath.append(os.path.join(data_root, person, line[0]))
                    label = line[1:]
                    label.append(str(np.squeeze(calibration["height_mm"])))
                    label.append(str(np.squeeze(calibration["height_pixel"])))
                    label.append(str(np.squeeze(calibration["width_mm"])))
                    label.append(str(np.squeeze(calibration["width_pixel"])))
                    self.labels.append(label)
                    self.monitor_pose.append([monitorPose["rvects"], monitorPose["tvecs"]])

    def single_image(self, index):
        img = cv2.imread(self.imgpath[index])
        height, width, _ = img.shape
        img = np.array(img)
        label = self.labels[index]

        gt_x = int(label[0])
        gt_y = int(label[1])
        h_mm, h_pixel, w_mm, w_pixel = float(label[-4]), float(label[-3]), float(label[-2]), float(label[-1])
        gt = (int(round(gt_x / w_pixel * self.res[0])), int(round(gt_y / h_pixel * self.res[1])))
        gt = (self.res[0] - gt[0], gt[1])   # Horizontally flip the gt, make the mirrored image seem more comfortable

        img = cv2.resize(img, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_NEAREST)
        leftEye, rightEye, face = get_box(label[2:], shift=False)
        leftEye = adjust_box(height, width, leftEye, self.resize)
        rightEye = adjust_box(height, width, rightEye, self.resize)
        face = adjust_box(height, width, face, self.resize)
        # print(leftEye, rightEye, face)
        cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), color=(0, 255, 0), thickness=1)
        cv2.rectangle(img, (leftEye[0], leftEye[1]), (leftEye[2], leftEye[3]), color=(0, 255, 0), thickness=1)
        cv2.rectangle(img, (rightEye[0], rightEye[1]), (rightEye[2], rightEye[3]), color=(0, 255, 0), thickness=1)

        rvec, tmat = self.monitor_pose[index]
        rmat = cv2.Rodrigues(rvec)[0]
        origin = np.array(label[20:23]).astype(np.float32)
        gaze_direction = np.array(label[23:26]).astype(np.float32) - origin
        projection = Gaze3DTo2D(gaze_direction, origin, rmat, tmat, require3d=False)
        project = (int(np.round(projection[0] / w_mm * self.res[0])), int(np.round(projection[1] / h_mm * self.res[1])))
        project = (self.res[0] - project[0], project[1])  # Horizontally flip the gt, make the mirrored image seem more comfortable
        return img, gt, (leftEye, rightEye, face), project

    def raw_image(self):
        for index in range(len(self.imgpath)):
            img, gt, _, project = self.single_image(index)
            background = np.full((self.res[1], self.res[0], 3), fill_value=255, dtype=np.uint8)
            h, w, _ = img.shape
            start_x, start_y = int(self.res[0] - w), 0
            background[start_y:start_y+h, start_x:start_x+w] = img
            img = background
            cv2.circle(img, center=gt, radius=15, color=(0, 0, 255), thickness=2)
            cv2.circle(img, center=project, radius=10, color=(255, 0, 114), thickness=2)
            print(gt, project)
            cv2.imshow('face', img)
            cv2.waitKey(0)


if __name__ == "__main__":
    data = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train"
    ShowImage(data).raw_image()

