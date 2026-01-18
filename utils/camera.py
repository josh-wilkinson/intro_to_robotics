import cv2
import subprocess
import threading
import numpy as np
import yaml
from rich import print
from timeit import default_timer as timer
import queue as Queue


class Camera:
    def __init__(self, exposure_time, gain) -> None:

        self.DEVICE = "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0"
        self.IM_WIDTH = 848 # New camera: 640
        self.IM_HEIGHT = 480
        self.FPS = 30 # 5, 10, 15, 20, 25, 30

        # we can use a larger image, but opencv detectmarkers will get really slow!
        # self.IM_WIDTH = 1920
        # self.IM_HEIGHT = 1080
        # self.FPS = 30

        self.exposure_time = exposure_time # between 1 and 5000
        self.gain = gain # between 0 and 100

        self.cap = None

        # self.queue = Queue.LifoQueue() # thread safe
        self.queue = None
        self.is_running = False

        with open("../camera_intrinsics.yml", "r") as stream:
            try:
                camera_intrinsics = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.CAMERA_MATRIX = np.array(camera_intrinsics["camera_matrix"]["data"]).reshape((camera_intrinsics["camera_matrix"]["rows"], camera_intrinsics["camera_matrix"]["cols"]))

        self.DIST_COEFFS = np.array(camera_intrinsics["distortion_coefficients"]["data"]).reshape((camera_intrinsics["distortion_coefficients"]["rows"], camera_intrinsics["distortion_coefficients"]["cols"]))

        # self.exposure = 400 # higher is brighter

        self.set_camera_properties(self.exposure_time, self.gain)

        self.time_last = timer()

        self.warm_up()

        # self.auto_set_brightness()

        # we run the camera read frames in a thread to always get the latest frame
        self.camera_thread = threading.Thread(target=self.worker_thread)
        self.start()


    def set_camera_properties(self, exposure_time=157, gain=50):
        """
        $ v4l2-ctl --all
        brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
        contrast 0x00980901 (int)    : min=0 max=64 step=1 default=22 value=22
        saturation 0x00980902 (int)    : min=0 max=128 step=1 default=64 value=64
        hue 0x00980903 (int)    : min=-40 max=40 step=1 default=0 value=0
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
        gamma 0x00980910 (int)    : min=72 max=500 step=1 default=110 value=110
        gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
        power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=1
                                        0: Disabled
                                        1: 50 Hz
                                        2: 60 Hz
        white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
        sharpness 0x0098091b (int)    : min=0 max=6 step=1 default=3 value=3
        backlight_compensation 0x0098091c (int)    : min=0 max=2 step=1 default=1 value=1

        Camera Controls

        auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
                                        1: Manual Mode
                                        3: Aperture Priority Mode
        exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
        exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
        """
        subprocess.call(f"v4l2-ctl -d {self.DEVICE} -c auto_exposure=1,exposure_time_absolute={exposure_time},gain={gain}", shell=True)

        self.cap = cv2.VideoCapture(self.DEVICE, cv2.CAP_V4L2)

        # get all output formats by running:
        # $ v4l2-ctl -d /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0 --list-formats-ext
        # note difference 'MJPG' (Motion-JPEG, compressed) and 'YUYV' (YUYV 4:2:2).
        # with MJPG we can get much higher fps

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.FPS)

    def worker_thread(self):
        while self.is_running:
            # get frame
            ret , frame = self.cap.read()

            time_now = timer()
            elapsed_time = time_now - self.time_last
            cam_fps = int(np.rint(1/elapsed_time))

            # self.queue.put((ret, frame, cam_fps, time_now))
            # we use a queue because it is thread safe
            self.queue = (ret, frame, cam_fps, time_now)

            self.time_last = time_now

            if cam_fps < 10:
                print(f"[red] cam_fps {cam_fps}")


    def read(self):
        if self.queue is not None:
            ret, frame, cam_fps, img_created = self.queue
            self.queue = None
            return ret, cv2.flip(frame, -1), cam_fps, img_created
        # if self.queue.qsize():
        #     ret, frame, cam_fps, img_created = self.queue.get()
        #     with self.queue.mutex:
        #         self.queue.queue.clear()
        #     return ret, cv2.flip(frame, -1), cam_fps, img_created
        else:
            return None, None, None, None


    def warm_up(self):
        self.ret , self.frame = self.cap.read()
        self.ret , self.frame = self.cap.read()


    def auto_set_exposure(self):
        brightness = self.get_brightness(self.frame)

        print("brightness", brightness)

        exposure_time = self.exposure_time
        while brightness < 100:
            exposure_time += 200

            print(f"[blue]increasing exposure to {self.exposure_time}")

            self.close()
            self.set_camera_properties(exposure_time)
            self.exposure_time = exposure_time
            self.start()
            self.warm_up()
            brightness = self.get_brightness(self.frame)

            print("brightness", brightness)


    def get_brightness(self, img):
        # https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv
        if len(img.shape) == 3:
            # Colored RGB or BGR (*Do Not* use HSV images with this function)
            # create brightness with euclidean norm
            return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            return np.average(img)


    def get_brightness_parameters(self, image, clip_hist_percent=10):
        """
        https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
        Perform contast enhancement of the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        self.auto_brightness_alpha = alpha
        self.auto_brightness_beta = beta

        print(f"[blue]auto brightness: alpha:{alpha}, beta:{beta}")

    def apply_brightness_params(self, img):
        if self.auto_brightness_alpha and self.auto_brightness_beta is not None:
            auto_result = cv2.convertScaleAbs(img, alpha=self.auto_brightness_alpha, beta=self.auto_brightness_beta)
            return auto_result
        return img

    def close(self):
        print("[blue]camera closing...")
        self.is_running = False

        self.camera_thread.join() # wait for thread to finish

        self.cap.release()
        print("[blue]camera closed")


    def start(self):
        self.is_running = True
        self.camera_thread.start()


if __name__ == '__main__':
    camera = Camera()
