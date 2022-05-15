import sys
import cv2
import numpy as np
import random
import datetime
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from scipy.io import savemat

import argparse


DarkGreyBlue = QColor(45, 45, 50)


def generate_click_label():
    return "L" if np.random.rand() < 0.5 else "R"


class TargetCircle(QWidget):
    def __init__(self, resolution, parent=None):
        super(TargetCircle, self).__init__(parent)
        self.font_size = 20
        self.circle_size = 40
        self.res = (resolution[0] - self.circle_size, resolution[1] - self.circle_size)  # prevent circles out of screen
        self.text_offset = [self.circle_size / 2 - self.font_size / 3, self.circle_size / 2 + self.font_size / 2]
        self.font = 'SansSerif'
        self.parent = parent

        self.circle_pivots, self.circle_count = self.generate_circle_pivot()    # Initialize circle positions
        self.generate_circle()  # Initialize a circle

        self.timer = QTimer()
        self.time_count()   # Start to countdown

    def generate_circle_pivot(self):    # Generate the circles by grids
        grids = 2, 2
        circle_per_grid = 2
        total = grids[0] * grids[1] * circle_per_grid
        assert self.res[0] % grids[0] == 0
        assert self.res[1] % grids[1] == 0
        grid_w = int(self.res[0] / grids[0])
        grid_h = int(self.res[1] / grids[1])
        pivots = []
        for i in range(grids[0]):
            for j in range(grids[1]):
                x1 = grid_w * i
                y1 = grid_h * j
                x2 = x1 + grid_w
                y2 = y1 + grid_h
                for _ in range(circle_per_grid):    # Generate <circle_per_grid> circles in every grid
                    x = random.randint(x1, x2)
                    y = random.randint(y1, y2)
                    pivots.append([x, y])
        return pivots, total

    def generate_circle(self):  # Randomly generate a new circle
        print(len(self.circle_pivots))
        try:
            self.circle_pivot = random.choice(self.circle_pivots)
        except IndexError:
            self.parent.Quit()  # When out of circles, call parent's quit function
        self.rgb = QColor(random.randint(170, 220), random.randint(170, 220), random.randint(170, 220))
        self.click_label = generate_click_label()

    def get_circle(self):   # Circle center
        center_x = self.circle_pivot[0] + self.circle_size / 2
        center_y = self.circle_pivot[1] + self.circle_size / 2
        return int(center_x), int(center_y), self.circle_size

    def get_click_label(self):
        return self.click_label

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        self.draw_circle(painter)
        painter.end()

    def draw_circle(self, painter):    # Main paint event to draw a circle
        x, y = self.circle_pivot
        painter.setPen(self.rgb)
        painter.setBrush(self.rgb)
        painter.drawEllipse(x, y, self.circle_size, self.circle_size)
        painter.setPen(DarkGreyBlue)
        painter.setFont(QFont(self.font, self.font_size))
        painter.drawText(x + self.text_offset[0], y + self.text_offset[1], self.click_label)

    def update_circle_timeout(self):    # Update circle due to timeout
        self.generate_circle()
        self.update()

    def update_circle_click(self):  # Update circle due to correct click
        self.circle_pivots.remove(self.circle_pivot)
        self.generate_circle()
        self.update()

    def time_count(self):
        self.timer.start(3000)
        self.timer.timeout.connect(self.update_circle_timeout)


class MainWindow(QWidget):
    def __init__(self, save_path, participant, resolution=(1920, 1080), parent=None):
        super(MainWindow, self).__init__(parent)
        # self.init_ui()
        self.circle = TargetCircle(resolution, self)    # Pass this Class as a parent to child Widget.
        self.participant = participant
        self.save_path = save_path
        self.capture = cv2.VideoCapture(0)
        self.frame = None
        self.camera_timer = QTimer()
        self.camera_timer.start(5)
        self.camera_timer.timeout.connect(self.activate_camera)
        # self.setWindowOpacity(0.9)  # Opacity

        palette = QPalette()
        palette.setColor(QPalette.Background, DarkGreyBlue)
        self.setPalette(palette)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.circle)

    def init_ui(self):
        self.label_mouse_x = QLabel(self)
        self.label_mouse_x.setGeometry(55, 5, 180, 30)
        self.label_mouse_x.setText('x')
        self.label_mouse_x.setMouseTracking(True)

        self.label_mouse_y = QLabel(self)
        self.label_mouse_y.setGeometry(55, 45, 180, 30)
        self.label_mouse_y.setText('y')
        self.label_mouse_y.setMouseTracking(True)

    def circle_button(self):
        self.btn2 = QPushButton("", self)
        self.btn2.setGeometry(255, 500, 100, 100)
        self.btn2.setStyleSheet("border-radius: 50; border: 1px; background-color: rgb(192, 192, 192); border-style")

    def Quit(self):
        self.close()
        self.capture.release()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        pos = event.windowPos()
        self.setMouseTracking(True)
        x = pos.x()
        y = pos.y()
        cx, cy, _ = self.circle.get_circle()
        if event.buttons() == Qt.LeftButton:
            # self.label_mouse_x.setText('Left  ' + str(x))
            # self.label_mouse_y.setText('Left  ' + str(y))
            if self.circle.get_click_label() == "L":
                if self.in_circle(x, y):
                    self.circle.update_circle_click()
                    self.circle.time_count()
                    self.save_data(cx, cy)

        elif event.buttons() == Qt.RightButton:
            # self.label_mouse_x.setText('Right  ' + str(x))
            # self.label_mouse_y.setText('Right  ' + str(y))
            if self.circle.get_click_label() == "R":
                if self.in_circle(x, y):
                    self.circle.update_circle_click()
                    self.circle.time_count()
                    self.save_data(cx, cy)

    def in_circle(self, x, y):
        cx, cy, size = self.circle.get_circle()
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if distance <= (size / 2):
            return True
        else:
            return False

    def save_data(self, cx, cy):
        now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')[:-3] + '.jpg'
        if self.frame is not None:
            if not os.path.exists(os.path.join(self.save_path, self.participant, "images")):
                os.makedirs(os.path.join(self.save_path, self.participant, "images"))
            img_path = os.path.join(self.participant, "images", now)
            label_path = os.path.join(self.save_path, self.participant, self.participant + ".txt")
            label_line = "images/" + now + " " + str(cx) + " " + str(cy)
            cv2.imwrite(os.path.join(self.save_path, img_path), self.frame)
            print(os.path.join(self.save_path, img_path))
            with open(label_path, 'a') as f:
                f.writelines(label_line)
                f.writelines("\n")

    def activate_camera(self):
        ret, self.frame = self.capture.read()


def main(args):
    save_path = "./"
    resolution = save_screen_info(args.path, save_path, participant=args.name)
    app = QApplication(sys.argv)
    window = MainWindow(save_path, participant=args.name, resolution=resolution)
    window.showFullScreen()
    window.show()
    sys.exit(app.exec_())


def save_screen_info(path, save_path, participant):
    with open(path) as f:
        lines = f.readlines()
    info = (lines[0].strip().replace('(', ' ').replace(')', ' ').split())
    resolution = info[1].split('x')
    resolution = int(resolution[0]), int(resolution[1])
    size = info[3].split('x')
    size = int(size[0]), int(size[1])
    if not os.path.exists(os.path.join(save_path, participant, "Calibration")):
        os.makedirs(os.path.join(save_path, participant, "Calibration"))
    mat = {'height_mm': size[1],
           'height_pixel': resolution[1],
           'width_mm': size[0],
           'width_pixel': resolution[0]}
    savemat(os.path.join(save_path, participant, "Calibration", "screenSize.mat"), mat)
    return resolution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GazeEstimation')
    parser.add_argument('--path', default='./screenInfo.txt', help='Screen size info')
    parser.add_argument('--name', default=None, type=str, help='Participant name')
    args = parser.parse_args()
    while args.name is None or args.name == '':
        args.name = input("请输入姓名拼音或缩写:")
    main(args)
    