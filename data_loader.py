import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.io import loadmat

from utils_function import *
from img_preprocess import face_eye_preprocess, normalize_gaze_vector, gamma_transform
from params import face_model, LEYE_INDICES, REYE_INDICES, MOUTH_INDICES, NOSE_INDICES


transform = T.Compose([
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
    ])


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
    def __init__(self, data_root, augment=False):
        self.imgpath = []
        self.labels = []
        self.camera_matrices = []
        self.monitorPoses = []
        self.face_model3d = face_model
        self.augment = augment
        for root, dirs, files in os.walk(data_root):
            for label_file in (x for x in files if x.endswith('.txt')):
                label_path = os.path.join(root, label_file)
                camera_matrix = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["cameraMatrix"],
                                         dtype=np.float64)
                distcoeff = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["distCoeffs"],
                                     dtype=np.float64)
                camera_participant = {"camera_matrix": camera_matrix, "dist_coefficients": distcoeff}

                # Use 2d to compute 3d. Exist errors in dataset?
                # monitorPose = os.path.join(root, "Calibration", "monitorPose.mat")
                # if os.path.exists(monitorPose):
                #     monitorPose = {"rvec": np.array(loadmat(monitorPose)["rvects"], dtype=np.float64),
                #                    "tvec": np.array(loadmat(monitorPose)["tvecs"], dtype=np.float64)
                #                    }
                # else:
                #     monitorPose = 0

                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.rstrip().split()
                        self.imgpath.append(os.path.join(root, line[0]))
                        label = line[1:]
                        self.labels.append(label)
                        self.camera_matrices.append(camera_participant)
                        # self.monitorPoses.append(monitorPose)

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = cv2.imread(self.imgpath[index])
        height, width, _ = img.shape
        if self.augment:
            r = np.random.rand()
            if r < 0.2:
                img = gamma_transform(img, gamma=0.5)
            elif 0.2 < r < 0.4:
                img = gamma_transform(img, gamma=2.0)

        img = np.array(img)
        label = self.labels[index]

        pixel_gt = np.array(label[:2]).astype(np.float32)
        gaze_vector = np.array(label[23:26]).astype(np.float32)
        # if isinstance(self.monitorPoses[index], dict):
        #     point_label = np.array(label[:2] + ["0"]).astype(np.float32).reshape((3, 1))
        #     rvec = self.monitorPoses[index]["rvec"]
        #     tvec = self.monitorPoses[index]["tvec"]
        #     rot_matrix, _ = cv2.Rodrigues(rvec)
        #     rot_inv = np.linalg.inv(rot_matrix)
        #     gaze_vector = np.dot(rot_inv, (point_label - tvec)).squeeze()
        fc = np.array(label[20:23]).astype(np.float32)
        headrotvectors = np.array(label[-6:-3]).astype(np.float32)
        headtransvectors = np.array(label[-3:]).astype(np.float32)
        # lefteye = np.array(label[27:31]).astype(np.float32)
        # righteye = np.array(label[31:35]).astype(np.float32)
        # mouth = np.array(label[10:14]).astype(np.float32)
        # nose = np.array(label[35:49]).astype(np.float32)
        # landmarks = np.vstack([lefteye, righteye, mouth])

        rot_matrix, _ = cv2.Rodrigues(headrotvectors)

        normalized_face_img, normalized_lEye_img, normalized_rEye_img, normalizing_rot = \
            face_eye_preprocess(img, self.face_model3d, self.camera_matrices[index],
                                rot_matrix, headtransvectors)

        # face_img = normalized_face_img / 255
        # face_img = face_img.transpose(2, 0, 1)
        face_img = transform(normalized_face_img)

        leftEye_img = normalized_lEye_img[:, :, np.newaxis]
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        im_right = cv2.flip(normalized_rEye_img, 1)    # H-flip right eye
        rightEye_img = im_right[:, :, np.newaxis]
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        gaze_vector = gaze_vector - fc
        normalized_head_rot = normalizing_rot @ rot_matrix
        normalized_gaze = normalize_gaze_vector(gaze_vector, normalizing_rot)
        normalized_headvector, _ = cv2.Rodrigues(normalized_head_rot)
        normalized_headvector = normalized_headvector.squeeze()
        normalized_yawpitch = vector_to_yawpitch(normalized_gaze)

        return torch.from_numpy(normalized_yawpitch).type(torch.FloatTensor), \
               torch.from_numpy(leftEye_img).type(torch.FloatTensor), \
               torch.from_numpy(rightEye_img).type(torch.FloatTensor), \
               face_img.type(torch.FloatTensor), \
               torch.from_numpy(normalized_headvector).type(torch.FloatTensor), \
                pixel_gt


# face_img.type(torch.FloatTensor), \
# torch.from_numpy(face_img).type(torch.FloatTensor), \





