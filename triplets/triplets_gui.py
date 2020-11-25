
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
import math
import time
import tkinter as tk
from PIL import ImageTk, Image
from threading import Thread
from landmarks_video import Video, LandmarksVideo, color_pallete


'''
A widget for displaying animation.
root: parent of widget
frames: list of frames to animate, each frame is a numpy array.
n_frames: number of frames to show in the animation - if less then length of frames, discard frames from beginning
    and end as needed. if more, pad with same frame in beginning and end.
'''
class Animation(tk.Canvas):
    def __init__(self, root, frames, n_frames=None, fps=30, *args, **kwargs):
        self.n_frames = len(frames)
        self.interval = 1 / fps
        self.root = root
        if 'width' in kwargs:
            width = kwargs['width']
            height = kwargs['height']
        else:
            height, width, *_ = frames[0].shape
        tk.Canvas.__init__(self, root, width=width, height=height, *args)
        self.n_frames = n_frames if n_frames else len(frames)
        if self.n_frames < len(frames):
            n_frames_to_discard = len(frames) - self.n_frames
            n_frames_to_discard_beginning = math.floor(n_frames_to_discard / 2)
            n_frames_to_discard_end = math.ceil(n_frames_to_discard / 2)
            frames = frames[n_frames_to_discard_beginning: - n_frames_to_discard_end]
        self.images = [ImageTk.PhotoImage(Image.fromarray(frame).resize((width, height))) for frame in frames]
        n_pad = self.n_frames - len(frames)
        self.pad_beginning, self.pad_ending = math.floor(n_pad / 2), math.ceil(n_pad / 2)
        self.images = [self.images[0]] * self.pad_beginning + self.images + [self.images[-1]] * self.pad_ending
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
    clips = [video[1000*i: 1000*i+ 3000 + 200*i: 10] for i in range(3, 7)]
    # anim1 = AnimationCanvas(root, video[10000:11000:2], width=180, height=180)
    # anim2 = AnimationCanvas(root, video[10000:11000:2], width=180, height=180)
    anims = [Animation(root, clips[i], n_frames=120, fps=60, width=250, height=250) for i in range(4)]
    # anim1.pack()
    # anim2.pack()
    root.mainloop()


if __name__ == '__main__':
    __main__()
