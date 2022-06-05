import os
from random import random
import torch
from torchvision import transforms as T
import torch.utils.data as data
import cv2
import mediapipe as mp
import numpy as np
from scipy.io import loadmat

from params import face_model, REYE_INDICES, LEYE_INDICES, MOUTH_INDICES
from params import camera as camera_local
from params import normalized_camera as normalized_camera_local
from models_3d import AFFNet as affnet_3d
from models_resnet import GazeResNet as gaze_resnet_model
from utils_function import load_model


transform = T.Compose([
        T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
    ])


def detect_landmarks_mediapipe(frame):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    detected_pts = []
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                pts = np.array([[pt.x * w, pt.y * h] for pt in face.landmark], dtype=np.float64)
                detected_pts.append(pts[:-10])
        return detected_pts


def estimate_head_pose_wrt_ccs(face_xy, camera, face_model3d):
    """Estimate the head pose by fitting 3D template model."""
    # If the number of the template points is small, cv2.solvePnP
    # becomes unstable, so set the default value for rvec and tvec
    # and set useExtrinsicGuess to True.
    # The default values of rvec and tvec below mean that the
    # initial estimate of the head pose is not rotated and the
    # face is in front of the camera.
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.array([0, 0, 1], dtype=np.float64)
    _, rvec, tvec = cv2.solvePnP(np.array(face_model3d, dtype=np.float64),
                                 face_xy,
                                 camera["camera_matrix"],
                                 camera["dist_coefficients"],
                                 rvec,
                                 tvec,
                                 useExtrinsicGuess=True,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    rot_matrix, _ = cv2.Rodrigues(rvec)
    return rot_matrix, tvec.squeeze()


def compute_head_xyz_in_ccs(rot_matrix, tvec, face_model3d):
    """Compute the transformed model."""
    face_xyz_ccs = face_model3d @ rot_matrix.T + tvec  # W.C.S --> C.C.S
    return face_xyz_ccs


def compute_face_eye_centers(face_xyz_ccs):
    """Compute the centers of the face and eyes.

    In the case of MPIIFaceGaze, the face center is defined as the
    average coordinates of the six points at the corners of both
    eyes and the mouth. In the case of ETH-XGaze, it's defined as
    the average coordinates of the six points at the corners of both
    eyes and the nose. The eye centers are defined as the average
    coordinates of the corners of each eye.
    """
    face_xyz_ccs_center = face_xyz_ccs[np.concatenate(
        [REYE_INDICES, LEYE_INDICES,
         MOUTH_INDICES])].mean(axis=0)
    reye_xyz_ccs_center = face_xyz_ccs[REYE_INDICES].mean(axis=0)
    leye_xyz_ccs_center = face_xyz_ccs[LEYE_INDICES].mean(axis=0)

    return face_xyz_ccs_center, leye_xyz_ccs_center, reye_xyz_ccs_center


def _normalize_image(image, face_normalizing_rot, distance, camera, normalized_camera):
    camera_matrix_inv = np.linalg.inv(camera["camera_matrix"])
    normalized_camera_matrix = normalized_camera["camera_matrix"]

    scale = _get_scale_matrix(distance)
    conversion_matrix = scale @ face_normalizing_rot

    projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

    normalized_image = cv2.warpPerspective(
        image, projection_matrix,
        (224, 224))
    return normalized_image


def _get_scale_matrix(distance):
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.6 / distance],
    ], dtype=np.float64)


def _compute_normalizing_rotation(face_xyz_ccs_center, rot_matrix):
    z_axis = _normalize_vector(face_xyz_ccs_center.ravel())
    head_x_axis = rot_matrix[:, 0]
    y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
    x_axis = _normalize_vector(np.cross(y_axis, z_axis))
    # rotation_vec, _ = cv2.Rodrigues(np.vstack([x_axis, y_axis, z_axis]))
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
    return rotation_matrix


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def eye_preprocess(eye_img):
    assert eye_img.shape == (224, 224, 3)
    start, end = 70, 154
    # Random shift
    random_disturb = np.random.randint(-20, 20)
    start, end = start + random_disturb, end + random_disturb

    eye_img = eye_img[start:end, start:end]
    normalized_image = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.equalizeHist(normalized_image)
    normalized_image = cv2.resize(normalized_image, (112, 112), interpolation=cv2.INTER_NEAREST)
    return normalized_image


def face_eye_preprocess(frame, face_model3d, camera, rot_matrix, tvec):
    assert rot_matrix.shape == (3, 3)
    assert tvec.shape == (3, )
    face_xyz_ccs = compute_head_xyz_in_ccs(rot_matrix, tvec, face_model3d)
    face_xyz_ccs_center, leye_xyz_ccs_center, reye_xyz_ccs_center = compute_face_eye_centers(face_xyz_ccs)

    # Normalize
    # ================================================================================
    face_normalizing_rot = _compute_normalizing_rotation(face_xyz_ccs_center, rot_matrix)
    leye_normalizing_rot = _compute_normalizing_rotation(leye_xyz_ccs_center, rot_matrix)
    reye_normalizing_rot = _compute_normalizing_rotation(reye_xyz_ccs_center, rot_matrix)

    normalized_face_image = _normalize_image(frame, face_normalizing_rot, np.linalg.norm(face_xyz_ccs_center), camera, normalized_camera_local)
    normalized_leye_image = _normalize_image(frame, leye_normalizing_rot, np.linalg.norm(leye_xyz_ccs_center), camera, normalized_camera_local)
    normalized_reye_image = _normalize_image(frame, reye_normalizing_rot, np.linalg.norm(reye_xyz_ccs_center), camera, normalized_camera_local)

    normalized_leye_image = eye_preprocess(normalized_leye_image)
    normalized_reye_image = eye_preprocess(normalized_reye_image)
    # ================================================================================

    return normalized_face_image, normalized_leye_image, normalized_reye_image, face_normalizing_rot


def normalize_gaze_vector(gaze_vector, normalizing_rot):
    """
    Here gaze_vector is a column vector
    """
    gaze_vector = gaze_vector.squeeze()
    normalized_gaze_vector = normalizing_rot @ gaze_vector
    return normalized_gaze_vector


class FaceNormalizer:
    def __init__(self, face_model3d, camera_params, normalized_camera_params, phase='test'):
        self.face_model3d = face_model3d
        self.camera = camera_params
        self.normalized_camera = normalized_camera_params
        self.phase = phase

    def normalize(self, pts, frame, indices=None, rot_matrix=None, tvec=None):
        if rot_matrix is None or tvec is None:
            if indices is None:
                indices = list(range(468))
            self.estimate_head_pose_wrt_ccs(pts, indices)
        else:
            self.rot_matrix, self.tvec = rot_matrix, tvec
        assert self.rot_matrix.shape == (3, 3)
        assert self.tvec.shape == (3,)
        face_xyz_ccs = self.compute_head_xyz_in_ccs()
        if self.phase == 'test':
            self.compute_face_eye_centers(face_xyz_ccs)
        else:  # Train
            self.compute_face_eye_centers(face_xyz_ccs)

        # Normalize
        # ================================================================================
        self._compute_normalizing_rotation()
        self._normalize_image(frame)
        # print('normalizing rot', face_normalizing_rot)
        normalized_head_rotation_matrix = self.face_normalizing_rotation_matrix @ self.rot_matrix
        # print('normal head rot', normalized_head_rot)
        self.normalized_headvector, _ = cv2.Rodrigues(normalized_head_rotation_matrix)
        self.normalized_headvector = self.normalized_headvector.squeeze()
        # print(normalized_headvector)
        # ================================================================================
        return self.face_normalized_image, self.leye_normalized_image, self.reye_normalized_image

    def _normalize_image(self, frame):
        camera_matrix_inv = np.linalg.inv(self.camera["camera_matrix"])
        normalized_camera_matrix = self.normalized_camera["camera_matrix"]

        distance = np.linalg.norm(self.face_xyz_ccs_center)
        scale = _get_scale_matrix(distance)
        conversion_matrix = scale @ self.face_normalizing_rotation_matrix
        projection_matrix_face = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        distance = np.linalg.norm(self.leye_xyz_ccs_center)
        scale = _get_scale_matrix(distance)
        conversion_matrix = scale @ self.leye_normalizing_rotation_matrix
        projection_matrix_leye = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        distance = np.linalg.norm(self.reye_xyz_ccs_center)
        scale = _get_scale_matrix(distance)
        conversion_matrix = scale @ self.reye_normalizing_rotation_matrix
        projection_matrix_reye = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv

        self.face_normalized_image = cv2.warpPerspective(frame, projection_matrix_face, (224, 224))
        self.leye_normalized_image = self.eye_preprocess(cv2.warpPerspective(
            frame, projection_matrix_leye, (224, 224)))
        self.reye_normalized_image = self.eye_preprocess(cv2.warpPerspective(
            frame, projection_matrix_reye, (224, 224)))

    def compute_face_eye_centers(self, face_xyz_ccs):
        """Compute the centers of the face and eyes.

        In the case of MPIIFaceGaze, the face center is defined as the
        average coordinates of the six points at the corners of both
        eyes and the mouth. In the case of ETH-XGaze, it's defined as
        the average coordinates of the six points at the corners of both
        eyes and the nose. The eye centers are defined as the average
        coordinates of the corners of each eye.
        """
        if self.phase == 'test':
            self.face_xyz_ccs_center = face_xyz_ccs[np.concatenate(
                [REYE_INDICES, LEYE_INDICES,
                 MOUTH_INDICES])].mean(axis=0)
            self.reye_xyz_ccs_center = face_xyz_ccs[REYE_INDICES].mean(axis=0)
            self.leye_xyz_ccs_center = face_xyz_ccs[LEYE_INDICES].mean(axis=0)
        else:  # Train
            self.face_xyz_ccs_center = face_xyz_ccs.mean(axis=0)
            self.leye_xyz_ccs_center = face_xyz_ccs[[0, 1]].mean(axis=0)
            self.reye_xyz_ccs_center = face_xyz_ccs[[2, 3]].mean(axis=0)

    def _compute_normalizing_rotation(self):
        z_axis = _normalize_vector(self.face_xyz_ccs_center.ravel())
        head_x_axis = self.rot_matrix[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        # rotation_vec, _ = cv2.Rodrigues(np.vstack([x_axis, y_axis, z_axis]))
        self.face_normalizing_rotation_matrix = np.vstack([x_axis, y_axis, z_axis])

        z_axis = _normalize_vector(self.leye_xyz_ccs_center.ravel())
        head_x_axis = self.rot_matrix[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        # rotation_vec, _ = cv2.Rodrigues(np.vstack([x_axis, y_axis, z_axis]))
        self.leye_normalizing_rotation_matrix = np.vstack([x_axis, y_axis, z_axis])

        z_axis = _normalize_vector(self.reye_xyz_ccs_center.ravel())
        head_x_axis = self.rot_matrix[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        # rotation_vec, _ = cv2.Rodrigues(np.vstack([x_axis, y_axis, z_axis]))
        self.reye_normalizing_rotation_matrix = np.vstack([x_axis, y_axis, z_axis])

    def eye_preprocess(self, eye_img):
        assert eye_img.shape == (224, 224, 3)
        start, end = 70, 154
        if self.phase == 'train':
            random_disturb = np.random.randint(-20, 20)
            start, end = start + random_disturb, end + random_disturb
        eye_img = eye_img[start:end, start:end]
        normalized_image = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        normalized_image = cv2.equalizeHist(normalized_image)
        normalized_image = cv2.resize(normalized_image, (112, 112), interpolation=cv2.INTER_NEAREST)
        return normalized_image

    def estimate_head_pose_wrt_ccs(self, face_xy, indices):
        """Estimate the head pose by fitting 3D template model."""
        # If the number of the template points is small, cv2.solvePnP
        # becomes unstable, so set the default value for rvec and tvec
        # and set useExtrinsicGuess to True.
        # The default values of rvec and tvec below mean that the
        # initial estimate of the head pose is not rotated and the
        # face is in front of the camera.
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.array([0, 0, 1], dtype=np.float64)
        _, rvec, tvec = cv2.solvePnP(self.face_model3d[indices],
                                     face_xy[indices],
                                     self.camera["camera_matrix"],
                                     self.camera["dist_coefficients"],
                                     rvec,
                                     tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        self.rot_matrix, _ = cv2.Rodrigues(rvec)
        self.tvec = tvec.squeeze()

    def compute_head_xyz_in_ccs(self):
        """Compute the transformed model."""
        face_xyz_ccs = self.face_model3d @ self.rot_matrix.T + self.tvec  # W.C.S --> C.C.S
        return face_xyz_ccs


class GazeEstimator(FaceNormalizer):
    def __init__(self, face_model3d, camera_params, normalized_camera_params, device):
        super(GazeEstimator, self).__init__(face_model3d, camera_params, normalized_camera_params, phase='test')
        self.device = device

    def get_predict_input(self, pts, frame, indices=None):
        normalized_face_img, normalized_lEye_img, normalized_rEye_img = super().normalize(pts, frame, indices=indices)
        # face_img = normalized_face_img / 255
        # face_img = face_img.transpose(2, 0, 1)
        face_img = transform(normalized_face_img)

        leftEye_img = normalized_lEye_img[:, :, np.newaxis]
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        im_right = cv2.flip(normalized_rEye_img, 1)  # H-flip right eye
        rightEye_img = im_right[:, :, np.newaxis]
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        normalized_lEye_img = torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(self.device)
        normalized_rEye_img = torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(self.device)
        normalized_face_img = face_img.type(torch.FloatTensor).to(self.device)
        normalized_headvector = torch.from_numpy(self.normalized_headvector).type(torch.FloatTensor).to(self.device)
        normalized_headvector = normalized_headvector.squeeze()

        normalized_lEye_img = normalized_lEye_img.unsqueeze(0)
        normalized_rEye_img = normalized_rEye_img.unsqueeze(0)
        normalized_face_img = normalized_face_img.unsqueeze(0)
        normalized_headvector = normalized_headvector.unsqueeze(0)

        return normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector

    def denormalize_gaze_vector(self, normalized_gaze_vector):
        """
        Here gaze vector is a row vector, and rotation matrices are
        orthogonal, so multiplying the rotation matrix from the right is
        the same as multiplying the inverse of the rotation matrix to the
        column gaze vector from the left.
        """
        gaze_vector = normalized_gaze_vector @ self.face_normalizing_rotation_matrix
        return gaze_vector

    def gaze2point(self, gaze_vector):
        """
        x = gaze_vector[0] * t + center[0]
        y = gaze_vector[1] * t + center[1]
        z = gaze_vector[2] * t + center[2]
        solve it for z=0 :
        """
        gaze_vector = gaze_vector.squeeze()
        center = self.face_xyz_ccs_center * 1e3
        t = - center[2] / gaze_vector[2]
        x = gaze_vector[0] * t + center[0]
        y = gaze_vector[1] * t + center[1]
        return x, y

    def mm2px(self, point, width=1920, height=1080,
              width_mm=508, height_mm=286):
        x, y = point
        x = - x + width_mm / 2
        x = (x / width_mm) * width
        y = (y / height_mm) * height
        return round(x), round(y)


def px2mm(coords, width=1920, height=1080,
          width_mm=508, height_mm=286):
    x = (coords[0] / width) * width_mm
    x = - x + width_mm / 2
    y = (coords[1] / height) * height_mm
    return x, y


def gamma_transform(img, gamma=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illumination = hsv[..., 2] / 255
    illumination = np.power(illumination, gamma)
    v = np.clip(illumination * 255, a_min=0, a_max=255)
    hsv[..., 2] = v.astype(np.int32)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def img():
    from utils_function import yawpitch_to_vector, vector_to_yawpitch
    from params import camera, normalized_camera
    from params import face_model as face_model3d
    weights = 'weights/2022-05-23/epoch_3.pth'
    # weights_resnet = "weights/2022-05-11-resnet/epoch_50.pth"
    cpu = False
    device = torch.device("cuda:0")

    gaze_model = affnet_3d()
    load_model(gaze_model, weights, cpu)
    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    path = r"D:\code\gaze\data\train\p00/p00.txt"
    root = path.rsplit('/', 1)[0]

    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        label = line.strip().split()
        pic = label[0]
        img = cv2.imread(os.path.join(root, pic))
        label = label[1:]
        gaze_vector = np.array(label[23:26]).astype(np.float32)
        fc = np.array(label[20:23]).astype(np.float32)
        headrotvectors = np.array(label[-6:-3]).astype(np.float32)
        headtransvectors = np.array(label[-3:]).astype(np.float32)

        camera_matrix = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["cameraMatrix"],
                                 dtype=np.float64)
        distcoeff = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["distCoeffs"],
                             dtype=np.float64)
        camera_participant = {"camera_matrix": camera_matrix, "dist_coefficients": distcoeff}

        rot_matrix, _ = cv2.Rodrigues(headrotvectors)

        normalized_face_img, normalized_lEye_img, normalized_rEye_img, normalizing_rot = \
            face_eye_preprocess(img, face_model3d, camera_participant,
                                rot_matrix, headtransvectors)

        face_img = normalized_face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = normalized_lEye_img[:, :, np.newaxis]
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        im_right = cv2.flip(normalized_rEye_img, 1)  # H-flip right eye
        rightEye_img = im_right[:, :, np.newaxis]
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        gaze_vector = gaze_vector - fc
        normalized_head_rot = normalizing_rot @ rot_matrix
        normalized_gaze = normalize_gaze_vector(gaze_vector, normalizing_rot)
        normalized_headvector, _ = cv2.Rodrigues(normalized_head_rot)
        normalized_headvector = normalized_headvector.squeeze()
        normalized_yawpitch = vector_to_yawpitch(normalized_gaze)

        pts = detect_landmarks_mediapipe(img)

        gaze_estimate = gaze_model(torch.from_numpy(leftEye_img).type(torch.FloatTensor).unsqueeze(0).cuda(),
               torch.from_numpy(rightEye_img).type(torch.FloatTensor).unsqueeze(0).cuda(),
               torch.from_numpy(face_img).type(torch.FloatTensor).unsqueeze(0).cuda(),
               torch.from_numpy(normalized_headvector).type(torch.FloatTensor).unsqueeze(0).cuda())

        print(normalized_yawpitch, gaze_estimate)

        criterion = torch.nn.SmoothL1Loss(reduction='sum')
        loss = criterion(torch.tensor(normalized_yawpitch), gaze_estimate.clone().detach().cpu())
        print("loss", loss)

        cv2.imshow('1', normalized_face_img)
        cv2.waitKey(0)


def web_cam():
    from params import MOUTH_INDICES, LEYE_INDICES, REYE_INDICES
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            _, frame = cap.read()
            pts = detect_landmarks_mediapipe(frame)[0]
            nose = [168, 197, 5, 1, 97, 2, 328]
            pts = pts.astype(np.int32)
            for pt in pts[np.hstack((nose, MOUTH_INDICES, LEYE_INDICES, REYE_INDICES))]:
            # for pt in pts:
                cv2.circle(frame, pt, 1, (0, 0, 255), -1)
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1920, 1440), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('1', frame)
            cv2.waitKey(1)

            # normalizer = FaceNormalizer(face_model3d=face_model, camera_params=camera_local, normalized_camera_params=normalized_camera_local)
            # for pt in pts:
            #     normalized_face, normalized_leye, normalized_reye = normalizer.normalize(pt, frame)
            #     cv2.imshow('n', normalized_face)
            #     cv2.imshow('nl', normalized_leye)
            #     cv2.imshow("nr", normalized_reye)
            #     cv2.waitKey(5)


if __name__ == "__main__":
    # web_cam()
    # img()
    img = cv2.imread(r"D:\code\gaze\data\train\lxc2\images\20220526-134701630.jpg")
    for i in range(5, 20):
        gamma = i / 10
        img_t = gamma_transform(img, gamma)
        cv2.imshow(f'gamma={gamma}', img_t)
        cv2.waitKey(0)


