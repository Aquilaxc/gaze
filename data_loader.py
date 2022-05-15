import os
import cv2
from torch.utils.data import Dataset
from scipy.io import loadmat

from utils_function import *
from img_preprocess import face_eye_preprocess, normalize_gaze_vector
from params import face_model, LEYE_INDICES, REYE_INDICES, MOUTH_INDICES


class MPIIFaceGaze2D(Dataset):
    def __init__(self, data_root):
        self.imgpath = []
        self.labels = []
        for root, dirs, files in os.walk(data_root):
            for label_file in (x for x in files if x.endswith('.txt')):
                label_path = os.path.join(root, label_file)
                calibration = loadmat(os.path.join(root, "Calibration", "screenSize.mat"))
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.rstrip().split()
                        self.imgpath.append(os.path.join(root, line[0]))
                        label = line[1:]
                        label.append(str(np.squeeze(calibration["height_mm"])))
                        label.append(str(np.squeeze(calibration["height_pixel"])))
                        label.append(str(np.squeeze(calibration["width_mm"])))
                        label.append(str(np.squeeze(calibration["width_pixel"])))
                        self.labels.append(label)

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = cv2.imread(self.imgpath[index])
        height, width, _ = img.shape
        img = np.array(img)
        label = self.labels[index]

        gt_x = int(label[0])
        gt_y = int(label[1])
        h_mm, h_pixel, w_mm, w_pixel = float(label[-4]), float(label[-3]), float(label[-2]), float(label[-1])
        # Transfer pixel to physical distance
        gt_x = gt_x / w_pixel * w_mm
        gt_y = gt_y / h_pixel * h_mm
        gt = torch.tensor([gt_x, gt_y])

        leftEye, rightEye, face = get_box(label[2:])
        leftEye_img, rightEye_img, face_img, rects = input_preprocess(img, leftEye, rightEye, face)

        return gt.type(torch.FloatTensor), \
               leftEye_img, rightEye_img, face_img, rects


class MPIIFaceGaze3D(Dataset):
    def __init__(self, data_root):
        self.imgpath = []
        self.labels = []
        self.camera_matrices = []
        self.face_model3d = np.vstack([face_model[LEYE_INDICES], face_model[REYE_INDICES], face_model[MOUTH_INDICES]])
        for root, dirs, files in os.walk(data_root):
            for label_file in (x for x in files if x.endswith('.txt')):
                label_path = os.path.join(root, label_file)
                camera_matrix = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["cameraMatrix"],
                                         dtype=np.float64)
                distcoeff = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["distCoeffs"],
                                     dtype=np.float64)
                camera_participant = {"camera_matrix": camera_matrix, "dist_coefficients": distcoeff}
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.rstrip().split()
                        self.imgpath.append(os.path.join(root, line[0]))
                        label = line[1:]
                        self.labels.append(label)
                        self.camera_matrices.append(camera_participant)

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = cv2.imread(self.imgpath[index])
        height, width, _ = img.shape
        img = np.array(img)
        label = self.labels[index]

        gaze_vector = np.array(label[23:26]).astype(np.float32)
        fc = np.array(label[20:23]).astype(np.float32)
        headrotvectors = np.array(label[-6:-3]).astype(np.float32)
        headtransvectors = np.array(label[-3:]).astype(np.float32)
        lefteye = np.array(label[2:6]).astype(np.float32)
        righteye = np.array(label[6:10]).astype(np.float32)
        mouth = np.array(label[10:14]).astype(np.float32)
        landmarks = np.vstack([lefteye, righteye, mouth])

        rot_matrix, _ = cv2.Rodrigues(headrotvectors)

        normalized_face_img, normalized_lEye_img, normalized_rEye_img, normalizing_rot, _, _ = \
            face_eye_preprocess(landmarks, img, self.face_model3d, self.camera_matrices[index],
                                rot_matrix, headtransvectors, phase='train')

        face_img = normalized_face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = normalized_lEye_img[:, :, np.newaxis]
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        im_right = cv2.flip(normalized_rEye_img, 1)    # H-flip right eye
        rightEye_img = im_right[:, :, np.newaxis]
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        gaze_vector = gaze_vector - fc
        normalized_head_rot = rot_matrix @ normalizing_rot
        normalized_gaze = normalize_gaze_vector(gaze_vector, normalizing_rot)
        normalized_headvector, _ = cv2.Rodrigues(normalized_head_rot)
        normalized_headvector = normalized_headvector.squeeze()

        return torch.from_numpy(normalized_gaze).type(torch.FloatTensor), \
               torch.from_numpy(leftEye_img).type(torch.FloatTensor), \
               torch.from_numpy(rightEye_img).type(torch.FloatTensor), \
               torch.from_numpy(face_img).type(torch.FloatTensor), \
               torch.from_numpy(normalized_headvector).type(torch.FloatTensor)

