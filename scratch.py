import os
import shutil

import cv2
from scipy.io import loadmat, savemat


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


def get_landmarks(img):
    from img_preprocess import detect_landmarks_mediapipe
    pts = detect_landmarks_mediapipe(img)
    pt_ret = []
    eye_nose = [33, 133, 362, 263, 168, 197, 5, 1, 97, 2, 328]
    for pt in pts:
        pt_ret = pt
    return pt_ret


def get_rotvector(pts, camera):
    import numpy as np
    from params import face_model as face_model3d
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.array([0, 0, 1], dtype=np.float64)
    eye_nose = [33, 133, 362, 263, 168, 197, 5, 1, 97, 2, 328]
    _, rvec, tvec = cv2.solvePnP(face_model3d,
                                 pts,
                                 camera["cameraMatrix"],
                                 camera["distCoeffs"],
                                 rvec,
                                 tvec,
                                 useExtrinsicGuess=True,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    rvec = [str(i) for i in rvec.squeeze()]
    tvec = [str(j) for j in tvec.squeeze()]
    vec = rvec + tvec
    return vec


def label_add_landmarks_and_rotvector(path):
    from scipy.io import loadmat
    import numpy as np
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt') and file != 'readme.txt':
                mat = os.path.join(root, "Calibration", "Camera.mat")
                camera = loadmat(mat)
                new_lines = []
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    img = line.strip().split()[0]
                    img = os.path.join(root, img)
                    print(img)
                    img = cv2.imread(img)
                    landmarks = get_landmarks(img)
                    if landmarks == []:
                        continue
                    vec = get_rotvector(landmarks, camera)
                    landmarks = np.round(landmarks).astype(np.int32).ravel()
                    new_line = ""
                    # for l in landmarks:
                    #     new_line += " "
                    #     new_line += str(l)
                    for v in vec:
                        new_line += " "
                        new_line += str(v)
                    new_line += '\n'
                    new_line = line.strip() + new_line
                    new_lines.append(new_line)
                with open(os.path.join(root, file), 'w') as f:
                    f.writelines(new_lines)


def show_image(path):
    import numpy as np
    root = path.rsplit('\\', 1)[0]
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        img = os.path.join(root, line[0])
        ori_ldmks = line[3:15]
        mp_ldmks = line[28:40]
        img = cv2.imread(img)
        for pt in np.array(ori_ldmks).astype(np.int32).reshape((-1, 2)):
            cv2.circle(img, (pt[0], pt[1]), 2, (0, 255, 0), -1)
        for pt in np.array(mp_ldmks).astype(np.int32).reshape((-1, 2)):
            cv2.circle(img, (pt[0], pt[1]), 2, (0, 0, 255), -1)

        cv2.imshow('1', img)
        cv2.waitKey(0)


def show_mat():
    mat = r"CaptureData\lxc2\Calibration\Camera.mat"
    mat = loadmat(mat)
    print(mat)


def save_cameraMat():
    from params import camera
    mat = {
        'cameraMatrix': camera["camera_matrix"],
        'distCoeffs': camera["dist_coefficients"]
    }
    savemat(r"CaptureData\lxc2\Calibration\Camera.mat", mat)


def check_transformation_from_scs_to_ccs():
    import numpy as np
    txt = r"D:\code\gaze\data\train\p00\p00.txt"
    monitor = os.path.join(txt.rsplit('\\', 1)[0], "Calibration", "monitorPose.mat")
    monitor = loadmat(monitor)
    rvec = monitor["rvects"]
    tvec = monitor["tvecs"]
    Camera = loadmat(os.path.join(txt.rsplit('\\', 1)[0], "Calibration", "Camera.mat"))
    camera_matrix = Camera["cameraMatrix"]
    rot_mat, _ = cv2.Rodrigues(rvec)
    screenSize = loadmat(os.path.join(txt.rsplit('\\', 1)[0], "Calibration", "screenSize.mat"))
    height_mm = screenSize["height_mm"]
    height_pixel = screenSize["height_pixel"]
    width_pixel = screenSize["width_pixel"]
    width_mm = screenSize["width_mm"]
    print(height_pixel, height_mm, width_pixel, width_mm)
    with open(txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        scs = np.array(line[1:3] + ["0"]).astype(np.float64).reshape(-1, 1)
        ccs = np.array(line[24:27]).astype(np.float64).reshape(-1, 1)
        ccs_flip = np.multiply(ccs, np.array([-1, 1, 1]).reshape((3, 1)))
        print(ccs, ccs_flip)
        computed = (np.dot(rot_mat, ccs) + tvec).squeeze()
        computed_flip = (np.dot(rot_mat, ccs_flip) + tvec).squeeze()
        x = computed[0] / width_mm * width_pixel
        y = computed[1] / height_mm * height_pixel
        x_flip = computed_flip[0] / width_mm * width_pixel
        y_flip = computed_flip[1] / height_mm * height_pixel
        computed = np.array([x.squeeze(), y.squeeze(), computed[-1]])
        computed_flip = np.array([x_flip.squeeze(), y_flip.squeeze(), computed_flip[-1]])
        scs = scs.squeeze().astype(np.int32)
        err = scs - computed
        print("scs", scs)
        print("compute", computed)
        print("computed flip", computed_flip)
        print("add", computed_flip[0] + computed[0])
        print("error", err)


def check_distribution_of_calibration(txt):
    import numpy as np
    with open(txt, 'r') as f:
        lines = f.readlines()
    background = np.full((1440, 2560, 3), fill_value=255, dtype=np.uint8)
    for line in lines:
        line = line.strip().split()
        dot = np.array(line[1:3]).astype(np.int32)
        cv2.circle(background, center=dot, radius=5, color=(0, 0, 255), thickness=-1)

    resize = 0.5
    img = cv2.resize(background, None, fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('distribution', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # png2jpg()
    # edit_label()
    # label_add_landmarks("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/lxc.txt")
    # show_landmarks("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze/train/lxc/images/20220413-164913680.jpg")
    # edit_mat()
    # check_mat()
    # normalize_one_face()
    # mediapipeface()
    # split_data("/media/cv1254/DATA/EyeTracking/data/MPIIFaceGaze")
    # remove_redundant()
    # check_exp_smoothing()

    # label = r"D:\code\gaze\data"
    # label_add_landmarks_and_rotvector(label)
    # save_cameraMat()
    # show_mat()
    # label = r"D:\code\gaze\data\MPIIFaceGaze\p00\p00.txt"
    # show_image(label)

    # img = "D:\code\gaze\data\MPIIFaceGaze\p07\day33/0369.jpg"
    # img = cv2.imread(img)
    # l = get_landmarks(img)
    # print(l)
    # check_transformation_from_scs_to_ccs()
    check_distribution_of_calibration(r"D:\code\gaze\CaptureData\test\test.txt")

