import copy

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import datetime
import argparse

from scipy.io import loadmat

from models_2d import AFFNet as affnet_2d
from models_3d import AFFNet as affnet_3d
from models_resnet import GazeResNet as gaze_resnet_model
from test_image import load_model
from utils_function import crop_img, get_box, input_preprocess, exp_smooth

import mediapipe as mp
from params import face_model, camera, normalized_camera
from img_preprocess import detect_landmarks_mediapipe, face_eye_preprocess, denormalize_gaze_vector, gaze2point, GazeEstimator
from Kalman import KalmanFilter


cpu = False
weights = 'weights/2022-05-10am/epoch_30.pth'
# weights = 'weights/2022-05-11/epoch_29.pth'
# weights = 'weights/2022-05-11-resnet/epoch_50.pth'
resolution_show = (1920, 1080)
calibration = [286, 1080, 508, 1920]    # h_mm, h_pixel, w_mm, w_pixel


def predict_gaze(net, device, frame, landmarks, calibration):
    try:
        leftEye, rightEye, face = get_box(landmarks, shift=False)
        leftEye_img, rightEye_img, face_img, rects = input_preprocess(frame, leftEye, rightEye, face)
        leftEye_img, rightEye_img, face_img, rects = leftEye_img.to(device), rightEye_img.to(device), face_img.to(
            device), rects.to(device)
        leftEye_img = leftEye_img.unsqueeze(0)
        rightEye_img = rightEye_img.unsqueeze(0)
        face_img = face_img.unsqueeze(0)
        rects = rects.unsqueeze(0)

        time_start = time.time()
        with torch.no_grad():
            gaze = net(leftEye_img, rightEye_img, face_img, rects)
        print("GAZE inference time:", time.time() - time_start)
        gaze = gaze.squeeze()
        h_mm, h_pixel, w_mm, w_pixel = calibration
        print("gaze", gaze)
        gaze_predict = int(torch.round(gaze[0] / w_mm * resolution_show[0])), int(torch.round(gaze[1] / h_mm * resolution_show[1]))
        # gaze_predict = np.array(gaze.cpu())
        gaze_predict = np.clip(gaze_predict[0], 0, resolution_show[0]), np.clip(gaze_predict[1], 0, resolution_show[1])
        cv2.rectangle(frame, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, (int(leftEye[0]), int(leftEye[1])), (int(leftEye[2]), int(leftEye[3])), color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, (int(rightEye[0]), int(rightEye[1])), (int(rightEye[2]), int(rightEye[3])), color=(0, 255, 0), thickness=2)
    except:
        gaze_predict = None
    finally:
        return gaze_predict, frame


def draw_gaze(predict, frame, gt=None):
    resize = 0.5
    background = np.full((resolution_show[1], resolution_show[0], 3), fill_value=255, dtype=np.uint8)
    img = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    h, w, _ = img.shape
    # start_x, start_y = int((resolution_show[0] - w) / 2), int((resolution_show[1] - h) / 2)
    start_x, start_y = resolution_show[0] - w, 0
    background[start_y:start_y + h, start_x:start_x + w] = img
    img = background
    cv2.line(img, (960, 0), (960, 100), color=(255, 222, 222), thickness=3)
    if gt is not None:
        gt = [resolution_show[0] - gt[0], gt[1]]  # Horizontally flip the gt, make the mirrored image seem more comfortable
        cv2.circle(img, center=gt, radius=20, color=(0, 0, 255), thickness=2)
        cv2.putText(img, f"Ground Truth: ({gt[0]}, {gt[1]})", (20, 20), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
    # predict = [resolution_show[0] - predict[0], predict[1]]   # Horizontally Flip
    if predict is not None:
        colors = [(255, 0, 114), (114, 0, 255)]
        for i, pre in enumerate(predict):
            cv2.circle(img, center=pre, radius=10, color=colors[i], thickness=2)
            cv2.putText(img, f"Predict:      ({pre[0]}, {pre[1]})", (20, 50), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
    return img


def smooth_gaze(gaze_before, gaze_now, alpha=0.8):
    if gaze_before is None:
        gaze_predict = gaze_now
    else:
        gaze_predict = exp_smooth(gaze_before, gaze_now, alpha)
    gaze_before = gaze_now
    gaze_predict = int(round(gaze_predict[0])), int(round(gaze_predict[1]))
    return gaze_before, gaze_predict



def mm2px(point, width_pixel=1920, height_pixel=1080, width_mm=508, height_mm=286):
    x, y = point
    x = - x + width_mm / 2
    x = (x / width_mm) * width_pixel
    y = (y / height_mm) * height_pixel
    return round(x), round(y)


def main():
    device = torch.device("cpu" if cpu else "cuda:0")
    cudnn.benchmark = True

    gaze_model = affnet_3d(res=False)
    # gaze_model = gaze_resnet_model()
    load_model(gaze_model, weights, cpu)
    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    capture = cv2.VideoCapture(0)

    log_x = []
    log_y = []
    xy_list = []

    P = loadmat("./CaptureData/lxc/projection.mat")['K']

    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            landmarks = detect_landmarks_mediapipe(frame)

            gaze_predictor = GazeEstimator(face_model3d=face_model, camera_params=camera, normalized_camera_params=normalized_camera, device=device)
            kf = KalmanFilter()
            for landmark in landmarks:
                normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector \
                    = gaze_predictor.get_predict_input(landmark, frame)
                print("normalized_headvector", normalized_headvector)
                gaze_predict = gaze_model(normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector)
                print("gaze_predict 1", gaze_predict)
                gaze_predict = gaze_predictor.denormalize_gaze_vector(gaze_predict)
                print("gaze_predict 3", gaze_predict)

                gaze_predict_point = gaze_predictor.gaze2point(gaze_predict)
                print("before 3", gaze_predict_point)
                # gaze_predict_point = gaze_predict_point - np.array([0, 5])
                print("after 3", gaze_predict_point)
                gaze_predict_point = mm2px(gaze_predict_point)

            if gaze_predict_point is not None:
                # gaze_before, gaze_predict = smooth_gaze(gaze_before, gaze_predict, alpha=0.3)

                x, y = gaze_predict_point
                xy_list.append([x, y])
                xy_list = xy_list[-10:]
                x, y = np.mean(xy_list, 0).astype(np.int)
                gaze_predict = x, y

                x1, y1 = kf.predict()
                x2, y2 = kf.update(gaze_predict)
                x2 = np.array(x2).squeeze().astype(np.int32)
                gaze_predict_Kalman = x2[0], x2[1]
                log_x.append(gaze_predict[0])
                log_y.append(gaze_predict[1])

            x, y = x2[0], x2[1]
            A = np.array([1, x, y, x**2, y**2, x * y, x**3, y**3, x**2 * y, x * y**2])
            gaze_predict_Kalman = np.dot(A, P).astype(np.int32).squeeze()

            print(gaze_predict_Kalman)
            out = draw_gaze([gaze_predict_Kalman], frame)
            out_win = "test"
            cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(out_win, out)
            if cv2.waitKey(1) & 0xff == 27:
                break
        capture.release()
        cv2.destroyAllWindows()
        print(f"max: ({max(log_x)}, {max(log_y)}), min: ({min(log_x)}, {min(log_y)}), mean: ({np.mean(log_x)}, {np.mean(log_y)})")
        dist = np.full((resolution_show[1], resolution_show[0], 3), fill_value=255, dtype=np.uint8)
        for i in range(len(log_x)):
            cv2.circle(dist, center=(log_x[i], log_y[i]), radius=10, color=(0, 0, 255), thickness=-1)
        save_dist_img = 'distribution/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
        # cv2.imwrite(save_dist_img, dist)


if __name__ == "__main__":
    main()
