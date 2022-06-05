import os
import cv2
import numpy as np
from scipy.io import loadmat

from params import face_model as face_model3d
from params import camera, normalized_camera, LEYE_INDICES, REYE_INDICES, MOUTH_INDICES
from img_preprocess import FaceNormalizer, detect_landmarks_mediapipe


face_normalizer = FaceNormalizer(face_model3d, camera, normalized_camera)


camera_y_offset = 35  # mm


def img_process(image):
    landmarks = detect_landmarks_mediapipe(image)
    for landmark in landmarks:
        face_img, leye_img, reye_img = face_normalizer.normalize(landmark, image)
        face_xyz_ccs_center = face_normalizer.face_xyz_ccs_center
        rot_matrix = face_normalizer.rot_matrix
        rvec, _ = cv2.Rodrigues(rot_matrix)
        tvec = face_normalizer.tvec
        return landmark[np.concatenate((LEYE_INDICES, REYE_INDICES, MOUTH_INDICES))].astype(np.int32).flatten(), \
               face_xyz_ccs_center, rvec.squeeze(), tvec


def gt_convert(xy, screenSize):
    xy = np.array(xy).astype(np.float64)
    x, y = xy[0], xy[1]
    width_pixel, width_mm = screenSize["width_pixel"].squeeze().astype(np.float64), \
                            screenSize["width_mm"].squeeze().astype(np.float64)
    height_pixel, height_mm = screenSize["height_pixel"].squeeze().astype(np.float64), \
                              screenSize["height_mm"].squeeze().astype(np.float64)
    x = width_pixel / 2 - x  # Move to the middle of screen
    x = x / width_pixel * width_mm
    y = y / height_pixel * height_mm - camera_y_offset
    z = 0
    return [x, y, z]


def gaze2p(gaze_vector, center):
    gaze_vector = gaze_vector.squeeze()
    center = center * 1e3
    t = - center[2] / gaze_vector[2]
    x = gaze_vector[0] * t + center[0]
    y = gaze_vector[1] * t + center[1]
    return x, y


def mm2px(point, width_pixel=1920, height_pixel=1080, width_mm=508, height_mm=286):
    x, y = point
    x = - x + width_mm / 2
    x = (x / width_mm) * width_pixel
    y = (y / height_mm) * height_pixel
    return round(x), round(y)


def read_and_write(label):
    root = label.rsplit('\\', 1)[0]
    screenSize = loadmat(os.path.join(root, "Calibration", "screenSize.mat"))
    print(screenSize)
    with open(label, 'r') as f:
        lines = f.readlines()
    lines_out = []
    for line in lines:
        old_line = line
        line = line.strip().split()
        assert len(line) < 4
        img = os.path.join(root, line[0])
        img = cv2.imread(img)
        img = cv2.flip(img, 1)
        landmark, fc, rvec, tvec = img_process(img)
        gt_ccs = gt_convert(line[1:], screenSize)
        new_line = " "
        for l in landmark:
            new_line += str(l)
            new_line += " "
        for r in rvec:
            new_line += str(r)
            new_line += " "
        for t in tvec:
            new_line += str(t)
            new_line += " "
        for c in fc:
            new_line += str(c * 1e3)
            new_line += " "
        for g in gt_ccs:
            new_line += str(g)
            new_line += " "
        for r in rvec:
            new_line += str(r)
            new_line += " "
        for t in tvec:
            new_line += str(t)
            new_line += " "
        new_line += "\n"
        line_out = old_line.strip() + new_line
        print(line_out)
        # print("length", len(line_out.strip().split()))
        # convert_p = gaze2p((gt_ccs - fc * 1e3), fc)
        # convert_p = mm2px(convert_p, width_pixel=2560, height_pixel=1440, width_mm=594, height_mm=333)
        # print(convert_p)
        lines_out.append(line_out)
    with open(label, 'w') as f:
        f.writelines(lines_out)


if __name__ == "__main__":
    txt = r"D:\code\gaze\CaptureData\cbj\cbj.txt"
    read_and_write(txt)
