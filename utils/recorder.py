import os, datetime
import cv2
import numpy as np
from rich import print
import json
import natsort


class Recorder:
    def __init__(self, dt) -> None:
        # TODO: implement recording and playback
        self.dt = dt
        self.recording = [] # list of [img, speed, turn]
        self.is_recording = False
        self.playback = False

        self.get_step = None
        

    def load_recording(self, recording_path):
        self.recording = [] # empty recording

        output = ""
        with open(os.path.join(recording_path, "movement.json"), "r+") as f:
            output = json.load(f)

        img_paths = os.listdir(recording_path)
        img_paths = natsort.os_sorted(img_paths)

        imgs = []

        for img_path in img_paths:
            if os.path.splitext(img_path)[1] == ".jpg":
                imgs.append(cv2.imread(os.path.join(recording_path, img_path)))
        
        if len(imgs) != len(output):
            print("[red]load recording failed! num images doesn't match!")

        for img, (speed, turn, x, y, theta, l, r) in zip(imgs, output):
            self.recording.append((img, speed, turn, x, y, theta, l, r))

        self.playback = True
        self.get_step = self._get_step()
    

    def save_recording(self):
        dir_name = os.path.join(os.getcwd(), "recordings", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if os.path.exists(dir_name):
            print(f"[red]{dir_name} already exists!")
            return
        
        os.makedirs(dir_name)
        # save speed, turn
        movements = [[speed, turn, np.round(x,3), np.round(y,3), np.round(theta, 3), np.round(l,3), np.round(r, 3)] for (_, speed, turn, x, y, theta, l, r) in self.recording]
        json_object = json.dumps(movements, indent=2)
        with open(os.path.join(dir_name, "movement.json"), "w") as f:
                f.write(json_object)

        for idx, (img, _, _, _, _, _, _, _) in enumerate(self.recording):
            img_filename = os.path.join(dir_name, f"img_{idx}.jpg")
            cv2.imwrite(img_filename, img)
        
        # TODO: save dt as well

        print("[green]Finished saving!")

    def start_recording(self):
        self.playback = False
        self.is_recording = True

    def stop_recording(self):
        self.playback = False
        self.is_recording = False
        self.save_recording()

    def save_step(self, img, speed, turn, x=0, y=0, theta=0, l=0, r=0):
        if self.is_recording:
            self.recording.append((img, speed, turn, x, y, theta, l, r))

    def _get_step(self):
        # TODO: implement generator
        count = 0
        while count < len(self.recording):
            print(f"count: {count}")
            yield self.recording[count]
            count += 1
        
