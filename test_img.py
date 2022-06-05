import copy
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import datetime
import argparse

import matplotlib.pyplot as plt

from scipy.io import loadmat

from models_2d import AFFNet as affnet_2d
from models_3d import AFFNet as affnet_3d
from models_resnet import GazeResNet as gaze_resnet_model
from utils_function import exp_smooth, load_model, yawpitch_to_vector

import mediapipe as mp
from params import face_model, camera, normalized_camera
from img_preprocess import detect_landmarks_mediapipe, GazeEstimator
from Kalman import KalmanFilter
from gaze_estimation_demo import get_resGaze_model as ResGaze


cpu = False
# weights = 'weights/2022-05-19/epoch_40.pth'
# weights = 'weights/2022-05-10am/epoch_30.pth'
# weights = 'weights/2022-05-24/epoch_45.pth'
weights = 'weights/2022-05-30/epoch_49.pth'
# weights = 'weights/2022-05-26/epoch_47.pth'



def draw_gaze(predict, frame, screen):
    resize = 0.2
    height_pixel, height_mm, width_pixel, width_mm = screen
    background = np.full((height_pixel, width_pixel, 3), fill_value=255, dtype=np.uint8)
    img = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    h, w, _ = img.shape
    start_x, start_y = width_pixel - w, 0
    background[start_y:start_y + h, start_x:start_x + w] = img
    img = background
    cv2.line(img, (int(width_pixel / 2), 0), (int(width_pixel / 2), 100), color=(255, 222, 222), thickness=3)

    if predict is not None:
        colors = [(255, 0, 0), (0, 0, 255)]
        text = ["Predict", "Ground Truth"]
        for i, pre in enumerate(predict):
            cv2.circle(img, center=pre, radius=10, color=colors[i], thickness=2)
            cv2.putText(img, f"{text[i]}: ({pre[0]}, {pre[1]})", (20, 50 * (i + 1)), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)

    return cv2.resize(img, (1600, 900), interpolation=cv2.INTER_NEAREST)


def mm2px(point, width_pixel=1920, height_pixel=1080, width_mm=508, height_mm=286):
    x, y = point
    x = - x + width_mm / 2
    x = (x / width_mm) * width_pixel
    y = (y / height_mm) * height_pixel
    return round(x), round(y)


def read_label(label):
    with open(label, 'r') as f:
        lines = f.readlines()
    images = []
    gt_in_pixel = []
    for line in lines:
        line = line.strip().split()
        images.append(os.path.join(label.rsplit(os.sep, 1)[0], line[0]))
        gt_in_pixel.append(line[1:3])
    screenSize = loadmat(os.path.join(label.rsplit(os.sep, 1)[0], "Calibration", "screenSize.mat"))
    screen = screenSize["height_pixel"].squeeze(), screenSize["height_mm"].squeeze(), \
             screenSize["width_pixel"].squeeze(), screenSize["width_mm"].squeeze()
    return images, np.array(gt_in_pixel).astype(np.float64), screen


def calculate_loss(predict, gt, screen):
    height_pixel, height_mm, width_pixel, width_mm = screen
    x_diff = (gt[0] - predict[0]) / width_pixel * width_mm
    y_diff = (gt[1] - gt[1]) / height_pixel * height_mm
    err = np.sqrt(x_diff ** 2 + y_diff ** 2)
    return err


def main(label):
    device = torch.device("cpu" if cpu else "cuda:0")
    cudnn.benchmark = True

    gaze_model = affnet_3d(res=False)
    load_model(gaze_model, weights, cpu)

    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    images, gt_in_pixel, screen = read_label(label)
    height_pixel, height_mm, width_pixel, width_mm = screen

    errors = []

    gazes = []

    Px = loadmat("./CaptureData/test/projection.mat")['Px']
    Py = loadmat("./CaptureData/test/projection.mat")['Py']

    for i, img in enumerate(images):
        frame = cv2.imread(img)
        landmarks = detect_landmarks_mediapipe(frame)
        gaze_predictor = GazeEstimator(face_model3d=face_model, camera_params=camera,
                                       normalized_camera_params=normalized_camera, device=device)
        for landmark in landmarks:
            normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector \
                = gaze_predictor.get_predict_input(landmark, frame)
            gaze_predict = gaze_model(normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector)

            # gaze_predict = yawpitch_to_vector(gaze_predict)

            gaze_predict = gaze_predictor.denormalize_gaze_vector(gaze_predict.cpu().detach().numpy())
            gaze_predict_point = gaze_predictor.gaze2point(gaze_predict)

            gaze_predict_point = gaze_predict_point[0], gaze_predict_point[1]
            gaze_predict_point = mm2px(gaze_predict_point, height_pixel=height_pixel, height_mm=height_mm,
                                       width_pixel=width_pixel, width_mm=width_mm)

            # =============================================================================
            w, h = 2560, 1440
            x, y = gaze_predict_point[0] / w, gaze_predict_point[1] / h
            # a = [x ** 3, y ** 3, x ** 2, y ** 2, x, y, 1]

            ax = [x**3, x**2, x, 1]
            ay = [y**3, y**2, y, 1]

            gaze_predict_x = np.dot(np.array(ax), Px.T).squeeze() * w
            gaze_predict_y = np.dot(np.array(ay), Py.T).squeeze() * h
            gaze_predict_point = gaze_predict_x, gaze_predict_y

            # gaze_predict_point = np.dot(np.array(a).reshape(1, -1), P).squeeze() * [w, h]
            gazes.append(list(gaze_predict_point))
            # =============================================================================



    print(gazes[:5])
    gazes = np.vstack(gazes)
    print(gazes.shape)
    x_gt = gt_in_pixel[:, 0].squeeze().astype(np.int32)
    y_gt = gt_in_pixel[:, 1].squeeze().astype(np.int32)
    x_pre = gazes[:, 0].squeeze()
    y_pre = gazes[:, 1].squeeze()

    from scipy.io import savemat
    x = {"gt": x_gt, "pred": x_pre}
    savemat("x_pred2.mat", x)

    plt.subplot(121)
    plt.scatter(x_gt, x_pre, s=5)
    line = list(range(np.max(x_gt)))
    plt.plot(line, line, color='red')

    plt.subplot(122)
    plt.scatter(y_gt, y_pre, s=5)
    line = list(range(np.max(y_gt)))
    plt.plot(line, line, color='red')
    plt.show()

        # print(gaze_predict_point)
        # out = draw_gaze([gaze_predict_point, gt_in_pixel[i]], frame, screen)
        # out_win = "test"
        # err = calculate_loss(gaze_predict_point, gt_in_pixel[i], screen)
        # print(err)
        # errors.append(err)

        # cv2.imshow(out_win, out)
        # if cv2.waitKey(0) & 0xff == 27:
        #     break
    # cv2.destroyAllWindows()
    # mean_error = np.mean(errors)
    # subject = label.rsplit(os.sep, 1)[-1][:-4]
    # print(f"{subject} error: {mean_error} mm\n")
    # return f"{subject} error: {mean_error} mm\n"




def all_error(dataset):
    lines = ""
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith(".txt"):
                label = os.path.join(root, file)
                line = main(label)
                lines += line
    print("=" * 40, " Result ", "=" * 40)
    print(lines)


if __name__ == "__main__":

    main(r"D:\code\gaze\CaptureData\lxc2\lxc2.txt")
    # main(r"D:\code\gaze\data\train\p14\p14.txt")
    # dataset = r"D:\code\gaze\data\train"
    # all_error(dataset)
