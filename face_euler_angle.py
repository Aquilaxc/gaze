import cv2
import numpy as np
import mediapipe as mp
import math
import glob
import os
from scipy.io import loadmat
from tqdm import tqdm
from params import face_model, camera


def test_camera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            h, w, _ = frame.shape
            mp_face_mesh = mp.solutions.face_mesh

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        all_landmarks = []
                        for index, landmark in enumerate(face.landmark):
                            one_landmark = []
                            one_landmark.append(landmark.x * w)
                            one_landmark.append(landmark.y * h)
                            # one_landmark.append(landmark.z)
                            all_landmarks.append(one_landmark)
                        all_landmarks = all_landmarks[:-10]
                rvec = np.zeros(3, dtype=np.float)
                tvec = np.array([0, 0, 1], dtype=np.float)
                facemodel = np.array(face_model, dtype=np.float64)
                alllandmarks = np.array(all_landmarks, dtype=np.float64)
                # print(facemodel.shape, alllandmarks.shape)
                _, rvec, tvec = cv2.solvePnP(facemodel,
                                             alllandmarks,
                                             camera["camera_matrix"],
                                             camera["dist_coefficients"],
                                             flags=cv2.SOLVEPNP_ITERATIVE
                                             )
                rotparams = GetEuler(rvec, tvec)
                rotparams = rotparams.squeeze().astype(np.int32)

                facecenter = np.mean(facemodel, axis=0)
                rmatrix, _ = cv2.Rodrigues(rvec)
                # print(rmatrix)
                frame = cv2.flip(frame, 1)
                cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame, f'yaw {rotparams[0]}, pitch {rotparams[1]}, roll {rotparams[2]}', (20, 20), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
                cv2.imshow('1', frame)
                # print(rvec, tvec)
                cv2.waitKey(5)


def GetEuler(rotation_vector, translation_vector):
    """
    此函数用于从旋转向量计算欧拉角
    :param rotation_vector: 输入为旋转向量
    :return: 返回欧拉角在三个轴上的值
    """
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
    yaw = eulerAngles[1]
    pitch = eulerAngles[0]
    roll = eulerAngles[2]
    rot_params = np.array([yaw, pitch, roll])
    return rot_params


def pic_face_rotvec(data_root):
    for txt in tqdm(glob.glob(os.path.join(data_root, "*/*.txt"))):
        with open(txt, 'r') as f:
            lines = f.readlines()
        root = txt.rsplit('/', 1)[0]
        camera_mtx = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["cameraMatrix"], dtype=np.float64)
        dist = np.array(loadmat(os.path.join(root, "Calibration", "Camera.mat"))["distCoeffs"], dtype=np.float64)
        new_lines = []
        # print(lines)
        for line in lines:
            line = line.strip()
            img = line.strip().split()[0]
            gt = line.strip().split()[15:21]
            # gt = np.array(gt).astype(np.float64)
            image = os.path.join(root, img)
            image = cv2.imread(image)
            h, w, _ = image.shape
            mp_face_mesh = mp.solutions.face_mesh

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                results = face_mesh.process(image)
                if results.multi_face_landmarks:
                    for face in results.multi_face_landmarks:
                        all_landmarks = []
                        for index, landmark in enumerate(face.landmark):
                            # if index in [33, 133, 362, 263, 78, 308]:
                                one_landmark = []
                                one_landmark.append(landmark.x * w)
                                one_landmark.append(landmark.y * h)
                                # one_landmark.append(landmark.z)
                                all_landmarks.append(one_landmark)
                                x = landmark.x
                                y = landmark.y
                                shape = image.shape
                                x_pixel = int(x * shape[1])
                                y_pixel = int(y * shape[0])
                                cv2.circle(image, (x_pixel, y_pixel), 1, (255, 0, 100), -1)
                        all_landmarks = all_landmarks[:-10]
                facemodel = np.array(face_model, dtype=np.float64)
                alllandmarks = np.array(all_landmarks, dtype=np.float64)
                _, rvec, tvec = cv2.solvePnP(facemodel,
                                             alllandmarks,
                                             camera_mtx,
                                             dist,
                                             flags=cv2.SOLVEPNP_ITERATIVE
                                             )

                rvec = [str(i) for i in rvec.squeeze()]
                tvec = [str(j) for j in tvec.squeeze()]
                vec = rvec + tvec
                for v in vec:
                        line = line + " " + v
            line = line + '\n'
            new_lines.append(line)
        with open(txt, 'w') as f:
            f.writelines(new_lines)


if __name__ == "__main__":
    test_camera()
    txt = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/test/"
    # pic_face_rotvec(txt)
