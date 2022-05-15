import os
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
import numpy as np
import cv2
import math
import time

# from tqdm import tqdm
from models_2d import AFFNet
from models_3d import AFFNet as affnet3d
from utils_function import get_box, input_preprocess


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    # if "state_dict" in pretrained_dict.keys():
    #     pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    # else:
    #     pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=True)
    return model


def read_label(label_path):

    imgpath = []
    labels = []
    data_root = label_path.rsplit('/', 1)[0]
    calibration_mat = loadmat(os.path.join(data_root, "Calibration", "screenSize.mat"))
    calibration = []
    calibration.append(np.squeeze(calibration_mat["height_mm"]))
    calibration.append(np.squeeze(calibration_mat["height_pixel"]))
    calibration.append(np.squeeze(calibration_mat["width_mm"]))
    calibration.append(np.squeeze(calibration_mat["width_pixel"]))
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split()
            imgpath.append(os.path.join(data_root, line[0]))
            label = line[1:]
            labels.append(label)
    return imgpath, labels, calibration


def caculate_test_error(label_path, weights, cpu=False):
    device = torch.device("cpu" if cpu else "cuda:0")
    cudnn.benchmark = True

    # Load Model
    gaze_model = AFFNet()
    load_model(gaze_model, weights, cpu)
    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    # Read labels
    imgpaths, labels, calibration = read_label(label_path)
    h_mm, h_pixel, w_mm, w_pixel = calibration

    error = 0
    time_total = 0

    for index, imgpath in enumerate(imgpaths):
        # Read image
        img = cv2.imread(imgpath)
        height, width, _ = img.shape
        img = np.array(img)
        label = labels[index]
        gt_pixel = int(label[0]), int(label[1])

        time_begin = time.time()

        # Predict
        leftEye, rightEye, face = get_box(label[2:], shift=False)
        leftEye_img, rightEye_img, face_img, rects = input_preprocess(img, leftEye, rightEye, face)
        leftEye_img, rightEye_img, face_img, rects = leftEye_img.to(device), rightEye_img.to(device), face_img.to(device), rects.to(device)
        leftEye_img = leftEye_img.unsqueeze(0)
        rightEye_img = rightEye_img.unsqueeze(0)
        face_img = face_img.unsqueeze(0)
        rects = rects.unsqueeze(0)
        gaze_mm = gaze_model(leftEye_img, rightEye_img, face_img, rects)
        gaze_mm = gaze_mm.squeeze()

        gt_mm = gt_pixel[0] / w_pixel * w_mm, gt_pixel[1] / h_pixel * h_mm

        err = math.sqrt((gaze_mm[0] - gt_mm[0]) ** 2 + (gaze_mm[1] - gt_mm[1]) ** 2)
        error += err
        time_total += (time.time() - time_begin)
        # print('gt ', gt_mm)
        # print('predict ', gaze_mm)
        # print('error ', err)
        # print('inference time', time.time() - time_begin)
    error /= len(imgpaths)
    time_total /= len(imgpaths)
    print(calibration)
    print('Average Error is {:.3f} cm, Average Inferece Time is {:.3f} s.'.format((error / 10), time_total))


def caculate_test_error_3d(test_path, weights, cpu=False):
    from data_loader import MPIIFaceGaze3D
    from tqdm import tqdm
    from gazehub.gazeconversion import Gaze3DTo2D
    device = torch.device("cpu" if cpu else "cuda:0")
    cudnn.benchmark = True

    rmat = loadmat(os.path.join("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/test/p00/Calibration/monitorPose.mat"))["rvects"]
    tmat = loadmat(os.path.join("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/test/p00/Calibration/monitorPose.mat"))["tvecs"]
    rmat = cv2.Rodrigues(rmat)[0]
    # Load Model
    gaze_model = affnet3d()
    load_model(gaze_model, weights, cpu)
    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    test_data = MPIIFaceGaze3D(test_path)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                shuffle=False)
    # criterion = torch.nn.SmoothL1Loss(reduction='sum')
    dataloader = tqdm(test_data)
    acc_loss = torch.zeros(1).to(device)
    length = len(dataloader)
    time_begin = time.time()
    for step, data in enumerate(dataloader):
        gt, leftEye_img, rightEye_img, face_img, rects, facecenter, gt_point = data
        gt, leftEye_img, rightEye_img, face_img, rects = gt.to(device), leftEye_img.to(device), rightEye_img.to(
            device), face_img.to(device), rects.to(device)
        # print(leftEye_img.size(), rightEye_img.size(), face_img.size(), rects.size())
        with torch.no_grad():
            gaze = gaze_model(leftEye_img, rightEye_img, face_img, rects)
        gaze = np.array(gaze.cpu().squeeze())
        facecenter = np.array(facecenter).squeeze()
        # print(gaze.shape)
        # print(facecenter)
        gaze_point = Gaze3DTo2D(gaze, facecenter, rmat, tmat)
        # loss = criterion(gaze, gt)
        print(gt_point)
        print(gaze_point)

    # print('Average Error is {:.3f} cm, Average Inferece Time is {:.3f} s.'.format((error / 10), time_total))


if __name__ == "__main__":
    test_path = '/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/test'
    weights = 'weights/2022-04-28/epoch_30.pth'
    caculate_test_error_3d(test_path, weights)
