import cv2
import numpy as np
import time
import os


mtx = np.array([[237.21036929,   0.,        479.18796748],
                [0.,         235.54517542, 366.09520559],
                [0.,           0.,           1.]],dtype=np.float64)

dist = np.array([[0.00978868],
                [-0.03383362],
                [0.03214306],
                [-0.00745617]],dtype=np.float64)


class Camera:
    def __init__(self, width=1640, height=1232, flip=2, disp_width=960, disp_height=720,
                 camera_matrix=mtx, camera_distortion=dist):
        self.__width = width
        self.__height = height
        self.__flip = flip
        self.__mtx = camera_matrix
        self.__dist = camera_distortion
        self.__dispW = disp_width
        self.__dispH = disp_height
        self.__gpu_mat = cv2.cuda_GpuMat()
        try:
            self.__mapx, self.__mapy = list(map(cv2.cuda_GpuMat, cv2.fisheye.initUndistortRectifyMap(
                self.__mtx, self.__dist, None, self.__mtx,
                (int(self.__dispH*1.5), int(self.__dispW*1.5)), 5)))
        except:
            self.__mapx, self.__mapy = cv2.fisheye.initUndistortRectifyMap(
                self.__mtx, self.__dist, None, self.__mtx,
                (int(self.__dispH*1.5), int(self.__dispW*1.5)), 5)
        self.__image_counter = self.load_image_counter()

    def load_image_counter(self):
        counter_file_path = "image_counter.txt"
        if os.path.exists(counter_file_path):
            with open(counter_file_path, "r") as file:
                return int(file.read())
        return 0

    def save_image_counter(self):
        counter_file_path = "image_counter.txt"
        with open(counter_file_path, "w") as file:
            file.write(str(self.__image_counter))

    def save_image(self, frame):
        output_folder = "images"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_filename = f"img{self.__image_counter}.png"
        image_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        self.__image_counter += 1
        self.save_image_counter()

    @property
    def camset(self):
        return f'nvarguscamerasrc !  video/x-raw(memory:NVMM), width={self.__width}, height={self.__height},' \
               ' format=NV12, framerate=30/1 ! nvvidconv flip-method=' +\
               str(self.__flip) + ' ! video/x-raw, width=' + str(self.__dispW) + ', height=' + str(self.__dispH) + \
               ', format=BGRx !videoconvert ! video/x-raw, format=BGR ! appsink'

    def undistort(self, frame):
        try:
            self.__gpu_mat.upload(frame)
            undistorted = cv2.cuda.remap(
                self.__gpu_mat, self.__mapx, self.__mapy, cv2.INTER_LINEAR)
            cpu_undistorted_frame = undistorted.download()
            output = cpu_undistorted_frame[:self.__dispH, :self.__dispW]
        except:
            undistorted = cv2.remap(
                frame, self.__mapx, self.__mapy, cv2.INTER_LINEAR)
            output = undistorted[:self.__dispH, :self.__dispW]
        return output

    def videocapture(self, camset=camset):
        return cv2.VideoCapture(camset)


webcam = Camera()

cap = webcam.videocapture(webcam.camset)

if True:
    while True:
        t1 = time.time()
        ret, frame = cap.read()

        print(f'fps {1 / (time.time() - t1)}')
        if ret:
            cv2.imshow("camera", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                webcam.save_image(frame)

    cap.release()
    cv2.destroyAllWindows()
# by Aria 🐸
