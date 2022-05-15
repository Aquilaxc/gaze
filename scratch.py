import os
import shutil

import cv2
from scipy.io import loadmat


def edit_label():
    data_root = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/lxc.txt"
    new_lines = []
    with open(data_root, 'r') as f:
        lines = f.readlines()
    for line in lines:
        new_line = line.replace('lxc/', '')
        new_lines.append(new_line)
    with open(data_root, 'w') as f:
        f.writelines(new_lines)


def png2jpg():
    from PIL import Image
    input_folder = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/other/lxc"
    output_folder = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/other/lxc_jpg"
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.png')):
            old_img = os.path.join(root, filename)
            jpgname = filename.split(".")[0] + ".jpg"
            new_img = os.path.join(output_folder, jpgname)
            img = Image.open(old_img)
            img.save(new_img)


def label_add_landmarks(label_path):
    import cv2
    from multilandmark68.detect_and_inference import detect_one_frame as predict_landmarks
    with open(label_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if len(line.strip().split()) > 3:
            new_lines.append(line)
            continue
        data_root = label_path.rsplit("/", 1)[0]
        img_path = os.path.join(data_root, line.strip().split()[0])
        print(img_path)
        img = cv2.imread(img_path)
        landmarks = predict_landmarks(img)
        if landmarks is None:
            continue
        for landmark in landmarks:
            line = line.strip() + " " + str(landmark)
        line += "\n"
        new_lines.append(line)
    with open(label_path, 'w') as f:
        f.writelines(new_lines)


def show_landmarks(path):
    import cv2
    from multilandmark68.detect_and_inference import show_one_frame as predict_landmarks
    img = cv2.imread(path)
    img = predict_landmarks(img)
    print(img.shape)
    cv2.imshow("1", img)
    cv2.waitKey(0)


def edit_mat():
    from scipy.io import loadmat, savemat
    calibration = loadmat("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/Calibration/screenSize.mat")
    print(calibration["height_mm"], calibration["height_pixel"])
    calibration["height_mm"] = 286
    calibration["height_pixel"] = 1080
    calibration["width_mm"] = 508
    calibration["width_pixel"] = 1920
    savemat('/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/Calibration/screenSize.mat', calibration)
    calibration = loadmat("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/Calibration/screenSize.mat")
    print(calibration["height_mm"], calibration["height_pixel"])


def remove_redundant():
    img_path = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/images"
    label_path = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/lxc.txt"
    imgs = os.listdir(img_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img = line.rstrip().split()[0].split('/')[-1]
        imgs.remove(img)
    for img in imgs:
        img = os.path.join(img_path, img)
        os.remove(img)


def check_exp_smoothing(alpha=0.8):
    import matplotlib.pyplot as plt
    import numpy as np
    time = [10.51, 16.53, 21.13, 26.19, 29.33, 31.51, 33.45, 35.37, 37.26, 39.44, 42.01, 44.05, 45.50, 47.32, 49.19,
            51.02, 52.35, 54.44, 56.18, 57.59, 59.32, 61.06, 62.45, 64.19, 65.51, 67.23, 68.56, 70.31, 72.06, 73.41]
    smooth = []
    gap = []
    gap_avg = []
    for i, t in enumerate(time):
        if i == 0:
            gap.append(t)
        else:
            gap.append(time[i] - time[i-1])

    for i, t in enumerate(gap):
        avg = np.mean(gap[:i+1])
        gap_avg.append(avg)

    for i, t in enumerate(gap):
        if i == 0:
            smooth.append(t)
        else:
            s = alpha * t + (1 - alpha) * gap[i-1]
            smooth.append(s)

    x = list(range(len(gap)))
    print(gap)
    print(gap_avg)
    print(smooth)
    plt.plot(x, gap, x, gap_avg, x, smooth)
    plt.show()


def check_mat():
    path = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/p10/Calibration/monitorPose.mat"
    path2 = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/p10/Calibration/screenSize.mat"
    path3 = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/p10/Calibration/Camera.mat"
    mat1 = loadmat(path)
    mat2 = loadmat(path2)
    mat3 = loadmat(path3)
    print(mat1.keys())
    print(mat1['rvects'])
    print(mat1['tvecs'])
    print(mat2.keys())
    print(mat2['height_mm'])
    print(mat2['height_pixel'])
    print(mat2['width_mm'])
    print(mat2['width_pixel'])
    print(mat3.items())


def normalize_video():
    from gazehub import data_processing_core as dpc
    camera = [
        [629.71581342, 0., 318.45568235],
        [0., 630.12058134, 239.78815847],
        [0., 0., 1.]
    ]
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        while True:
            norm = dpc.norm(center=facecenter,
                            gazetarget=target,
                            headrotvec=headrotvectors,
                            imsize=(224, 224),
                            camparams=camera)

            # Get the normalized face image.
            im_face = norm.GetImage(img)
            cv2.imshow('1', im_face)
            cv2.waitKey(0)


def normalize_one_face():
    from gazehub import data_processing_core as dpc
    import numpy as np
    label = "/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/p06/p06.txt"
    data_root = label.rsplit("/", 1)[0]
    print(data_root)
    calib = loadmat(os.path.join(data_root, "Calibration", "Camera.mat"))
    camera = calib["cameraMatrix"]
    with open(label, "r") as f:
        lines = f.readlines()
    for line in lines:
        l = line.strip().split()
        img = cv2.imread(os.path.join(data_root, l[0]))
        print(img.shape)
        facecenter = np.array(l[21:24]).astype(np.float32)
        target = np.array(l[24:27]).astype(np.float32)
        headrotvectors = np.array(l[15:18]).astype(np.float32)
        lefteye_leftcorner = np.array(l[3:5]).astype(np.float32)
        lefteye_rightcorner = np.array(l[5:7]).astype(np.float32)
        righteye_leftcorner = np.array(l[7:9]).astype(np.float32)
        righteye_rightcorner = np.array(l[9:11]).astype(np.float32)
        norm = dpc.norm(center=facecenter,
                        gazetarget=target,
                        headrotvec=headrotvectors,
                        imsize=(224, 224),
                        camparams=camera)

        # Get the normalized face image.
        im_face = norm.GetImage(img)
        print(im_face.shape)
        cv2.imshow('1', im_face)
        cv2.waitKey(0)

        # Crop Right eye images.
        llc = norm.GetNewPos(lefteye_leftcorner)
        lrc = norm.GetNewPos(lefteye_rightcorner)
        im_left = norm.CropEye(llc, lrc)
        print(im_left.shape)
        im_left = dpc.EqualizeHist(im_left)
        print(im_left[:, :, np.newaxis].shape)
        cv2.imshow('2', im_left)
        cv2.waitKey(0)
        #
        # Crop Right eye images.
        rlc = norm.GetNewPos(righteye_leftcorner)
        rrc = norm.GetNewPos(righteye_rightcorner)
        im_right = norm.CropEye(rlc, rrc)
        im_right = dpc.EqualizeHist(im_right)
        cv2.imshow('3', im_right)
        cv2.waitKey(0)
        #
        # # Acquire related info
        # gaze = norm.GetGaze(scale)
        # head = norm.GetHeadRot(vector)
        # origin = norm.GetCoordinate(facecenter)
        # rvec, svec = norm.GetParams()


def mediapipeface():
    import numpy as np
    from gazehub import data_processing_core as dpc
    target = np.array([1, 1, 1])
    camera_matrix = [
        [632.70890028, 0., 317.32130435],
        [0., 632.83295582, 242.72876265],
        [0., 0., 1.]
    ]
    camera_matrix = np.array(camera_matrix)
    norm = dpc.norm(center=facecenter,
                    gazetarget=target,
                    headrotvec=headrotvectors,
                    imsize=(224, 224),
                    camparams=camera_matrix)


def split_data(path):
    train = os.path.join(path, "train")
    test = os.path.join(path, "test")
    for root, dirs, files in os.walk(train):
        for file in files:
            if file.endswith(".txt"):
                new_lines_train = []
                new_lines_test = []
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                new_lines_train += lines[:-100]
                new_lines_test += lines[-100:]
                for line in new_lines_test:
                    img = line.strip().split()[0]
                    ori_img = os.path.join(root, img)
                    dest_img = ori_img.replace('train', 'test')
                    dest = dest_img.rsplit('/', 1)[0]
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                    shutil.copy(ori_img, dest_img)
                    os.remove(ori_img)
                with open(os.path.join(root, file).replace('train', 'test'), 'w') as f:
                    f.writelines(new_lines_test)
                with open(os.path.join(root, file), 'w') as f:
                    f.writelines(new_lines_train)
                ori_calibration = os.path.join(root, "Calibration")
                for file in os.listdir(ori_calibration):
                    calib_file = os.path.join(ori_calibration, file)
                    dest_file = calib_file.replace('train', 'test')
                    dest = dest_file.rsplit('/', 1)[0]
                    if not os.path.exists(dest):
                        os.mkdir(dest)
                    shutil.copy(calib_file, dest_file)


if __name__ == "__main__":
    # png2jpg()
    # edit_label()
    # label_add_landmarks("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/lxc.txt")
    # show_landmarks("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/images/20220413-164913680.jpg")
    # edit_mat()
    # check_mat()
    # normalize_one_face()
    mediapipeface()
    # split_data("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze")
    # remove_redundant()
    # check_exp_smoothing()
