import sys
# sys.path.append('..')
import os
import numpy as np
import cv2 as cv
from PIL import Image
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
from deep_cluster.dataloader import LandmarkDataset, SequenceDataset
from matplotlib import pyplot as plt
from collections import Counter
import torch
from torch.utils.data import ConcatDataset
import pickle
import re
import math
import time
import tkinter as tk
from PIL import ImageTk, Image
from threading import Thread, Event
from multiprocessing import Process
from landmarks_video import Video, LandmarksVideo, color_pallete
from sample_triplets import Segment, triplets_segments_gen, load_segments
import math
from contextlib import contextmanager

@contextmanager
def timethis(label):
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"{label} took {elapsed} seconds")


def get_seg_clip(vid, seg, n_frames, fps=120):
    frames = vid[seg.start_frame: seg.start_frame + seg.n_frames: 240//fps]
    frames = list(frames)
    if n_frames < len(frames):
        n_frames_to_discard = len(frames) - n_frames
        n_frames_to_discard_beginning = math.floor(n_frames_to_discard / 2)
        n_frames_to_discard_end = math.ceil(n_frames_to_discard / 2)
        frames = frames[n_frames_to_discard_beginning: - n_frames_to_discard_end]
    n_pad = n_frames - len(frames)
    pad_beginning, pad_ending = math.floor(n_pad / 2), math.ceil(n_pad / 2)
    frames = [frames[0]] * pad_beginning + frames + [frames[-1]] * pad_ending
    return np.stack(frames)


def get_random_clips(vid, n_frames=240, max_n_clips=100, fps=60):
    for i in range(max_n_clips):
        random_idxs = np.random.randint(0, 3*10**5, size=(3,))
        print(vid[random_idxs[0]].shape)
        yield [vid[idx:idx + n_frames:240//fps] for idx in random_idxs]

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
        self.stop = Event()
        if 'width' in kwargs:
            width = kwargs['width']
            height = kwargs['height']
        else:
            height, width, *_ = frames[0].shape
            if 'rescale' in kwargs:
                height, width = int(height * kwargs['rescale']), int(width * kwargs['rescale'])
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
        # self.thread.setDaemon(True)
        self.root.after(0, self.thread.start)

    def animation(self):
        try:
            while not self.stop.is_set():
                for i in range(self.n_frames):
                    time.sleep(self.interval)
                    if self.stop.is_set():
                        return
                    self.create_image(0, 0, image=self.images[i], anchor='nw')
                    self.update()
        except tk.TclError as e:
            print("[INFO] caught a RuntimeError")

    def destroy(self):
        self.stop.set()
        self.thread.join(0.1)
        super().destroy()


class ClipsDisplay(tk.Frame):
    def __init__(self, root, clips, fps=30, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.window = tk.Frame(self)
        n_frames = int(np.mean([len(clip) for clip in clips]))
        self.anchor_anim = Animation(self, clips[0], fps=fps, n_frames=n_frames, rescale=2.)#, width=100, height=240)
        self.anchor_anim.pack(side=tk.LEFT)
        pos_frame = tk.Frame(self.window)
        neg_frame = tk.Frame(self.window)
        choice_var = tk.BooleanVar(self)
        radio_button_1 = tk.Radiobutton(pos_frame, var=choice_var, value=True)
        radio_button_2 = tk.Radiobutton(neg_frame, var=choice_var, value=False)
        self.pos_anim = Animation(pos_frame, clips[1], fps=fps, n_frames=n_frames, rescale=2.)#, width=100, height=200)
        self.neg_anim = Animation(neg_frame, clips[2], fps=fps, n_frames=n_frames, rescale=2.)#, width=100, height=200)
        self.pos_anim.pack()
        self.neg_anim.pack()
        radio_button_1.pack(side=tk.BOTTOM)
        radio_button_2.pack(side=tk.BOTTOM)
        pos_frame.pack(side=tk.LEFT)
        neg_frame.pack(side=tk.LEFT)
        self.window.pack()
        self.pack()

    # def destroy(self):
    #     self.anchor_anim.destroy()
    #     self.pos_anim.destroy()
    #     self.neg_anim.destroy()
    #     super().destroy()


class App(tk.Frame):
    def __init__(self, root, video, triplet_segment_gen, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.video = video
        self.triplets_gen = triplet_segment_gen
        self.display = tk.Frame()
        self.display.pack()
        self.next_button = tk.Button(self, command=self.reload_display, text="NEXT")
        self.next_button.pack(side=tk.BOTTOM)
        self.bind('<Return>', self.reload_display)
        self.bind('<space>', self.reload_display)
        self.bind('q', self.quit())
        self.focus_set()
        self.load_clips()
        self.reload_display()
        self.pack()

    def reload_display(self, event=None):
        self.display.destroy()
        self.display = ClipsDisplay(self, self.clips, fps=30)
        self.display.pack(side=tk.TOP)
        self.load_clips()


    def load_clips(self):
        print("start load clips")
        # anchor, pos, neg = next(self.triplets_gen)
        # self.clips = [get_seg_clip(self.video, seg, n_frames=60, fps=120) for seg in [anchor, pos, neg]]
        self.clips = next(self.triplets_gen)
        print("finish load clips")


data_root = Path("/home/orel/Storage/Data/K6/")
landmark_file = data_root/'2020-03-23'/'Down'/'0008DeepCut_resnet50_Down2May25shuffle1_1030000.h5'
video_file = data_root/'2020-03-23'/'Down'/'0008.MP4'


def __main__():
    root = tk.Tk()
    video = LandmarksVideo(data_root/'2020-03-23'/'Down', include_landmarks=False)
    app = App(root, video, get_random_clips(video, n_frames=120, fps=120))
    root.mainloop()


if __name__ == '__main__':
    __main__()
