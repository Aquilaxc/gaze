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
from utils_function import exp_smooth, load_model, yawpitch_to_vector

import mediapipe as mp
from params import face_model, camera, normalized_camera, LEYE_INDICES, REYE_INDICES, MOUTH_INDICES
from img_preprocess import detect_landmarks_mediapipe, GazeEstimator
from Kalman import KalmanFilter
from gaze_estimation_demo import get_resGaze_model as ResGaze


cpu = False
# weights = 'weights/2022-05-19/epoch_40.pth'
# weights = 'weights/2022-05-10am/epoch_30.pth'
# weights = 'weights/2022-05-26/epoch_47.pth'
# weights = 'weights/2022-05-24/epoch_45.pth'
weights = 'weights/2022-05-30/epoch_49.pth'
# weights = 'weights/2022-05-11/epoch_29.pth'
# weights = 'weights/2022-05-11-resnet/epoch_50.pth'
height_pixel = 1440
height_mm = 333
width_pixel = 2560
width_mm = 594


# def predict_gaze(net, device, frame, landmarks, calibration):
#     try:
#         leftEye, rightEye, face = get_box(landmarks, shift=False)
#         leftEye_img, rightEye_img, face_img, rects = input_preprocess(frame, leftEye, rightEye, face)
#         leftEye_img, rightEye_img, face_img, rects = leftEye_img.to(device), rightEye_img.to(device), face_img.to(
#             device), rects.to(device)
#         leftEye_img = leftEye_img.unsqueeze(0)
#         rightEye_img = rightEye_img.unsqueeze(0)
#         face_img = face_img.unsqueeze(0)
#         rects = rects.unsqueeze(0)
#
#         time_start = time.time()
#         with torch.no_grad():
#             gaze = net(leftEye_img, rightEye_img, face_img, rects)
#         print("GAZE inference time:", time.time() - time_start)
#         gaze = gaze.squeeze()
#         h_mm, h_pixel, w_mm, w_pixel = calibration
#         print("gaze", gaze)
#         gaze_predict = int(torch.round(gaze[0] / w_mm * resolution_show[0])), int(torch.round(gaze[1] / h_mm * resolution_show[1]))
#         # gaze_predict = np.array(gaze.cpu())
#         gaze_predict = np.clip(gaze_predict[0], 0, resolution_show[0]), np.clip(gaze_predict[1], 0, resolution_show[1])
#         cv2.rectangle(frame, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), color=(0, 255, 0), thickness=2)
#         cv2.rectangle(frame, (int(leftEye[0]), int(leftEye[1])), (int(leftEye[2]), int(leftEye[3])), color=(0, 255, 0), thickness=2)
#         cv2.rectangle(frame, (int(rightEye[0]), int(rightEye[1])), (int(rightEye[2]), int(rightEye[3])), color=(0, 255, 0), thickness=2)
#     except:
#         gaze_predict = None
#     finally:
#         return gaze_predict, frame


def draw_gaze(predict, frame):
    resize = 0.5
    background = np.full((height_pixel, width_pixel, 3), fill_value=45, dtype=np.uint8)
    img = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    h, w, _ = img.shape
    start_x, start_y = width_pixel - w, 0
    background[start_y:start_y + h, start_x:start_x + w] = img
    img = background
    cv2.line(img, (int(width_pixel / 2), 0), (int(width_pixel / 2), 100), color=(45, 22, 22), thickness=3)

    if predict is not None:
        colors = [(255, 255, 114), (114, 255, 255)]
        for i, pre in enumerate(predict):
            # cv2.circle(img, center=pre, radius=10, color=colors[i], thickness=2)
            # draw_fading_square(img, pre)
            draw_square(img, pre, square_size=(640, 480))
            cv2.putText(img, f"Predict:      ({pre[0]}, {pre[1]})", (20, 50), cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
    return img


def draw_gaze_arrow(face_center_xy, image_in, yawpitch, thickness=2, color=(255, 255, 0), sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    yawpitch = yawpitch.cpu().detach().numpy().squeeze()

    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = face_center_xy
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(yawpitch[1]) * np.cos(yawpitch[0])
    dy = -length * np.sin(yawpitch[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out


def draw_square(img, pre, square_size=(40, 40)):
    square_color = (222, 200, 200)
    line_color = (85, 65, 65)
    h, w, _ = img.shape
    x_max, y_max = w // square_size[0], h // square_size[1]
    x, y = pre[0] // square_size[0], pre[1] // square_size[1]
    for i in range(x_max):
        for j in range(y_max):
            cv2.rectangle(img,  (i * square_size[0], j * square_size[1]),
                          ((i + 1) * square_size[0], (j + 1) * square_size[1]),
                          line_color, 2)
    cv2.rectangle(img, (x * square_size[0], y * square_size[1]),
                  ((x + 1) * square_size[0], (y + 1) * square_size[1]),
                  square_color, -1)
    return img


def draw_fading_square(img, pre):
    square_size = 40
    h, w, _ = img.shape
    center_i, center_j = pre[0] // 40, pre[1] // 40
    offset = [-2, -1, 0, 1, 2]
    offset_x, offset_y = np.meshgrid(offset, offset)
    for x, y in zip(offset_x.flatten(), offset_y.flatten()):
        if np.abs(x) == 2 or np.abs(y) == 2:
            color = (200, 200, 255)
        elif np.abs(x) == 1 or np.abs(y) == 1:
            color = (120, 120, 255)
        else:
            color = (0, 0, 255)
        x += center_i
        y += center_j
        cv2.rectangle(img, (x * square_size, y * square_size), ((x + 1) * square_size, (y + 1) * square_size),
                      color, -1)
        cv2.rectangle(img, (x * square_size, y * square_size), ((x + 1) * square_size, (y + 1) * square_size),
                      (255, 255, 255), 3)
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
    # gaze_model = ResGaze()

    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    capture = cv2.VideoCapture(0)

    log_x = []
    log_y = []
    xy_list = []

    Px = loadmat("./CaptureData/test/projection.mat")['Px']
    Py = loadmat("./CaptureData/test/projection.mat")['Py']

    from params import LEYE_INDICES, REYE_INDICES, MOUTH_INDICES, NOSE_INDICES
    indices = np.hstack([LEYE_INDICES, REYE_INDICES, NOSE_INDICES])

    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            landmarks = detect_landmarks_mediapipe(frame)

            gaze_predictor = GazeEstimator(face_model3d=face_model, camera_params=camera, normalized_camera_params=normalized_camera, device=device)
            kf = KalmanFilter()
            for landmark in landmarks:
                normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector \
                    = gaze_predictor.get_predict_input(landmark, frame)
                gaze_predict = gaze_model(normalized_lEye_img, normalized_rEye_img, normalized_face_img, normalized_headvector)
                gaze_predict_yawpitch = gaze_predict
                print("angles", gaze_predict)
                # gaze_predict = yawpitch_to_vector(gaze_predict)
                print("vector", gaze_predict)
                gaze_predict = gaze_predictor.denormalize_gaze_vector(gaze_predict.cpu().detach().numpy())
                print("denorm vector", gaze_predict)
                gaze_predict_point = gaze_predictor.gaze2point(gaze_predict)



                # gaze_predict_point = np.array([gaze_predict_point[0], gaze_predict_point[1], 1])
                # camera_matrix = camera["camera_matrix"]
                # print('111', gaze_predict_point)
                # gaze_predict_point = np.dot(camera_matrix, gaze_predict_point)
                # print('222', gaze_predict_point)
                # gaze_predict_point = gaze_predict_point[:2]

                gaze_predict_point = gaze_predict_point[0], gaze_predict_point[1]
                gaze_predict_point = mm2px(gaze_predict_point, height_pixel=height_pixel, height_mm=height_mm,
                                           width_pixel=width_pixel, width_mm=width_mm)

                face_center = np.mean(landmark[np.concatenate((LEYE_INDICES, REYE_INDICES, MOUTH_INDICES))])


            h, w = 1440, 2560

            if gaze_predict_point is not None:
                # gaze_before, gaze_predict = smooth_gaze(gaze_before, gaze_predict, alpha=0.3)

                x, y = gaze_predict_point
                xy_list.append([x, y])
                xy_list = xy_list[-10:]
                x, y = np.mean(xy_list, 0).astype(np.int)
                x_orig = copy.deepcopy(x)

                x, y = x / w, y / h
                ax = [x ** 3, x ** 2, x, 1]
                ay = [y ** 3, y ** 2, y, 1]
                gaze_predict_x = np.dot(np.array(ax), Px.T).squeeze() * w
                gaze_predict_y = np.dot(np.array(ay), Py.T).squeeze() * h
                gaze_predict = gaze_predict_x.astype(np.int32), gaze_predict_y.astype(np.int32)

                # x1, y1 = kf.predict()
                # x2, y2 = kf.update(gaze_predict)
                # x2 = np.array(x2).squeeze().astype(np.int32)
                # gaze_predict_Kalman = x2[0], x2[1]
                # log_x.append(gaze_predict[0])
                # log_y.append(gaze_predict[1])

            print("P", Px)
            # gaze_predict = np.dot(np.array(gaze_predict).reshape(1, -1), P).squeeze()

            # gaze_predict = np.multiply(gaze_predict, [w, h]).astype(np.int32)
            # gaze_predict = [x_orig, gaze_predict[1]]

            out = draw_gaze([gaze_predict], frame)
            out_win = "test"
            cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(out_win, out)
            if cv2.waitKey(1) & 0xff == 27:
                break
        capture.release()
        cv2.destroyAllWindows()
        print(f"max: ({max(log_x)}, {max(log_y)}), min: ({min(log_x)}, {min(log_y)}), mean: ({np.mean(log_x)}, {np.mean(log_y)})")
        dist = np.full((height_pixel, width_pixel, 3), fill_value=255, dtype=np.uint8)
        for i in range(len(log_x)):
            cv2.circle(dist, center=(log_x[i], log_y[i]), radius=10, color=(0, 0, 255), thickness=-1)
        save_dist_img = 'distribution/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
        # cv2.imwrite(save_dist_img, dist)


if __name__ == "__main__":
    main()
