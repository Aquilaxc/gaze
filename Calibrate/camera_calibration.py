import cv2
import numpy as np
import glob
import datetime


def capture_photo():
    capture = cv2.VideoCapture(0)
    if capture.isOpened():
        count = 0
        while True:
            ret, frame = capture.read()
            cv2.imshow('1', frame)
            if cv2.waitKey(1) & 0xff == 32:  # Blank
                now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')[:-3] + '.jpg'
                file = "images/" + now
                cv2.imwrite(file, frame)
                count += 1
                print(file, count)
            if cv2.waitKey(1) & 0xff == 27:  # Esc
                capture.release()
                cv2.destroyAllWindows()
                break


def calibrate(rows=9, cols=9):
    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob("/media/cv1254/DATA/EyeTracking/code/GazeEstimate2d/Calibrate/images/*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        cv2.imshow('img', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
        print(ret)

        if ret:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (rows, cols), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            #print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.imshow("img", img)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

    print(len(img_points))
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    print("ret:", ret)
    print("mtx:\n", mtx) # 内参数矩阵


    print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
    print("tvecs:\n", tvecs ) # 平移向量  # 外参数

    print("-----------------------------------------------------")
    img = cv2.imread(images[2])


    h, w = img.shape[:2]
    print(h, w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h)) #显示更大范围的图片（正常重映射之后会删掉一部分图像）
    print (newcameramtx)
    dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
    x,y,w,h = roi
    dst1 = dst[y:y+h,x:x+w]
    cv2.imwrite('calibresult3.jpg', dst1)
    print ("dst的大小为:", dst1.shape)


def matrix_cached():
    mtx = [
        [632.70890028, 0., 317.32130435],
        [0., 632.83295582, 242.72876265],
        [0., 0., 1.]
    ]


if __name__ == "__main__":
#     capture_photo()
    calibrate()
