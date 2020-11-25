
import numpy as np
import cv2 as cv
from PIL import Image
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
from dataloader import LandmarkDataset, SequenceDataset
from matplotlib import pyplot as plt
import os
from collections import Counter
import torch
from torch.utils.data import ConcatDataset
import pickle
import re
import time
import tkinter as tk
from PIL import ImageTk, Image
from threading import Thread
from landmarks_video import Video, LandmarksVideo, color_pallete

'''
A widget for displaying animation.
root: parent of widget
frames: list of frames to animate, each frame is a numpy array.

'''
class Animation(tk.Canvas):
    def __init__(self, root, frames, n_frames=100, fps=30, *args, **kwargs):
        self.n_frames = len(frames)
        self.interval = 1 / fps
        self.root = root
        if 'width' in kwargs:
            width = kwargs['width']
            height = kwargs['height']
        else:
            height, width, *_ = frames[0].shape
        tk.Canvas.__init__(self, root, width=width, height=height, *args)
        self.n_frames = n_frames
        self.images = [ImageTk.PhotoImage(Image.fromarray(frame).resize((width, height))) for frame in frames]
        self.create_image(0, 0, image=self.images[0], anchor='nw')
        self.pack()
        self.thread = Thread(target=self.animation)
        self.root.after(0, self.thread.start)

    def animation(self):
        while True:
            for i in range(self.n_frames):
                time.sleep(self.interval)
                self.create_image(0, 0, image=self.images[i], anchor='nw')
                self.update()


data_root = Path("/home/orel/Storage/Data/K6/")
landmark_file = data_root/'2020-03-23'/'Down'/'0008DeepCut_resnet50_Down2May25shuffle1_1030000.h5'
video_file = data_root/'2020-03-23'/'Down'/'0008.MP4'




def __main__():
    root = tk.Tk()
    video = Video(video_file)
    clip = video[10000:11000:20]
    # anim1 = AnimationCanvas(root, video[10000:11000:2], width=180, height=180)
    # anim2 = AnimationCanvas(root, video[10000:11000:2], width=180, height=180)
    anims = [Animation(root, clip, fps=60, width=180, height=180) for i in range(4)]
    # anim1.pack()
    # anim2.pack()
    root.mainloop()


if __name__ == '__main__':
    __main__()
