import sys
# sys.path.append('..')
import os
import numpy as np
import cv2 as cv
from scipy import signal as sig
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
    frames = vid[seg.start_frame: seg.start_frame + seg.n_frames: int(math.ceil(vid.fps/fps))]
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


def get_random_clips(vid, duration=0.5, max_n_clips=100, fps=60):
    for i in range(max_n_clips):
        random_idxs = np.random.randint(0, 3*10**5, size=(3,))
        # print(vid[random_idxs[0]].shape)
        n_frames = int(duration * vid.fps)
        yield [vid[idx:idx + n_frames: int(math.ceil(vid.fps/fps))] for idx in random_idxs]

'''
A widget for displaying animation.
root: parent of widget
frames: list of frames to animate, each frame is a numpy array.
n_frames: number of frames to show in the animation - if less then length of frames, discard frames from beginning
    and end as needed. if more, pad with same frame in beginning and end.
'''
class Animation(tk.Canvas):
    def __init__(self, root, frames, n_frames=None, fps=30, *args, **kwargs):
        # self.n_frames = len(frames)
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
        self.anchor_anim = Animation(self, clips[0], fps=fps, n_frames=n_frames, rescale=1.5)#, width=100, height=240)
        self.anchor_anim.pack(side=tk.LEFT)
        pos_frame = tk.Frame(self.window)
        neg_frame = tk.Frame(self.window)
        choice_var = tk.IntVar(self)
        self.choice_var = choice_var
        radio_button_1 = tk.Radiobutton(pos_frame, var=choice_var, value=1)
        radio_button_2 = tk.Radiobutton(neg_frame, var=choice_var, value=2)
        self.pos_anim = Animation(pos_frame, clips[1], fps=fps, n_frames=n_frames, rescale=1.5)#, width=100, height=200)
        self.neg_anim = Animation(neg_frame, clips[2], fps=fps, n_frames=n_frames, rescale=1.5)#, width=100, height=200)
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

import pandas as pd
class App(tk.Frame):
    def __init__(self, root, video, encoded=None, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root
        self.encoded = encoded
        # self.df = pd.DataFrame(columns=['video_file', 'anchor', 'sample1', 'sample2', 'selected'], index=pd.Index(np.arange(100)))
        self.saved_triplets = []
        self.video = video
        # self.triplets_gen = triplet_segment_gen
        self.i = 0
        self.display = tk.Frame()
        self.display.pack()
        self.next_button = tk.Button(self, command=self.next, text="NEXT")
        self.next_button.pack(side=tk.BOTTOM)
        self.ilabel = tk.Label(self, text=str(self.i), font=("Courier", 32))
        self.ilabel.pack(side=tk.LEFT)
        if self.encoded is not None or True:
            self.dist_label1 = tk.Label(self, text='distance from anchor to positive:')
            self.dist_label1.pack(side=tk.BOTTOM)
            self.dist_label2 = tk.Label(self, text='distance from anchor to negative:')
            self.dist_label2.pack(side=tk.BOTTOM)
            self.dist_label3 = tk.Label(self, text='distance from negative to positive:')
            self.dist_label3.pack(side=tk.BOTTOM)
        self.bind('<Return>', lambda event: self.save())
        self.bind('<space>', self.next)
        self.bind('w', self.next)
        self.bind('q', lambda event: self.quit())
        self.focus_set()
        self.load_clips()
        self.reload_display()
        self.bind('<Left>', lambda event: self.display.choice_var.set(1))
        self.bind('<Right>', lambda event: self.display.choice_var.set(2))
        self.bind('<Down>', lambda event: self.display.choice_var.set(0))
        self.bind('a', lambda event: self.display.choice_var.set(1))
        self.bind('d', lambda event: self.display.choice_var.set(2))
        self.bind('s', lambda event: self.display.choice_var.set(0))
        self.bind('e', lambda event: self.save())
        self.pack()

    def sample_random_triplets(self, n_frames=120, fps=120):
        from scipy import signal as sig
        random_idxs = np.random.randint(len(self.video) - self.video.fps, size=3)
        while True:
            random_idxs = np.random.randint(len(self.video) - self.video.fps, size=3)
            self.segments = [slice(idx, idx + n_frames, int(math.ceil(self.video.fps/fps))) for idx in random_idxs]
            # break
            for seg in self.segments:
                # landmarks = self.video.landmarks[seg]
                data =self.video.landmarks[seg].T
                filt = sig.butter(4, 6, btype='low', output='ba', fs=self.video.fps)
                filtered = sig.filtfilt(*filt, data).T
                avg_diff = np.linalg.norm(np.diff(filtered, axis=0), axis=-1).mean()
                if avg_diff < 0.05:
                    break
            else:
                break

        # enc_segments = [slice(idx // 4 + 15, (idx + n_frames) // 4 - 15 + 1) for idx in random_idxs]
        if self.encoded is not None:
            self.clip_encodeings = [self.encoded[idx // int(len(self.video)/len(self.encoded))] for idx in random_idxs]
        # print(self.clip_encodeings[0].shape)
        self.clips = [self.video[seg] for seg in self.segments]

    def next(self, event=None):
        self.save_triplet()
        self.i += 1
        self.reload_display()

    def reload_display(self):
        self.display.destroy()
        self.display = ClipsDisplay(self, self.clips, fps=30)
        self.display.pack(side=tk.TOP)
        if self.encoded is not None:
            dist1 = np.linalg.norm(self.clip_encodeings[0] - self.clip_encodeings[1])
            dist2 = np.linalg.norm(self.clip_encodeings[0] - self.clip_encodeings[2])
            dist3 = np.linalg.norm(self.clip_encodeings[2] - self.clip_encodeings[1])
            print(dist1, dist2, dist3)
            # self.dist_label1['text'] = f"distance between anchor and sample1: {diffs[0]:.2f}"
            # self.dist_label2['text'] = f"distance between anchor and sample2: {diffs[1]:.2f}"
            # self.dist_label3['text'] = f"distance between sample2 and sample1: {diffs[2]:.2f}"
        diffs = []
        for i in range(3):
            seg = self.segments[i]
            data =self.video.landmarks[seg].T
            filt = sig.butter(4, 6, btype='low', output='ba', fs=self.video.fps)
            filtered = sig.filtfilt(*filt, data).T
            diffs.append(np.linalg.norm(np.diff(filtered, axis=0), axis=-1).mean())
        self.dist_label1['text'] = f"anchor: {diffs[0]:.4f}"
        self.dist_label2['text'] = f"sample 1: {diffs[1]:.4f}"
        self.dist_label3['text'] = f"sample2: {diffs[2]:.4f}"
        self.load_clips()

    def save_triplet(self):
        self.saved_triplets.append({'video_file': self.video.video_file,
                                    'anchor': (self.segments[0].start, self.segments[0].stop),
                                    'sample1': (self.segments[1].start, self.segments[1].stop),
                                    'sample2': (self.segments[2].start, self.segments[2].stop),
                                    'selected': self.display.choice_var.get()})

    def load_clips(self):
        print("start load clips")
        # anchor, pos, neg = next(self.triplets_gen)
        # self.clips = [get_seg_clip(self.video, seg, n_frames=60, fps=120) for seg in [anchor, pos, neg]]
        self.sample_random_triplets()
        self.ilabel['text'] = self.i
        print("finish load clips")

    def save(self):
        df = pd.DataFrame.from_records(self.saved_triplets)
        # df.to_json(path_or_buf='triplets/data/selected_triplets.json', orient='records')
        save_path = 'data/selected_triplets.csv'
        mode = 'a' if os.path.exists(save_path) else 'r'
        df.to_csv(path_or_buf=save_path, mode=mode)
        self.saved_triplets = []

    def quit(self):
        print("quitting")
        self.save()
        self.root.quit()


def decode_seg_string(seg_string):
    start, end = seg_string[1:-1].split(',')
    start, end = int(start), int(end)
    start_idx, end_idx = start , end 
    return (start_idx, end_idx)

class VerificationApp(tk.Frame):
    def __init__(self, root, video, df, encoded, to_save=True, start_idx=-1, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.root = root
        self.df = df
        self.to_save = to_save
        self.encoded = encoded
        # self.df = pd.DataFrame(columns=['video_file', 'anchor', 'sample1', 'sample2', 'selected'], index=pd.Index(np.arange(100)))
        self.saved_triplets = []
        self.video = video
        self.i = start_idx
        self.display = tk.Frame()
        self.display.pack()
        self.next_button = tk.Button(self, command=self.next, text="NEXT")
        self.next_button.pack(side=tk.BOTTOM)
        self.bind('<Return>', lambda event: self.save())
        self.bind('<space>', self.next)
        self.bind('w', self.next)
        self.bind('q', lambda event: self.quit())
        if self.encoded is not None:
            self.dist_label1 = tk.Label(self, text='distance from anchor to positive:')
            self.dist_label1.pack(side=tk.BOTTOM)
            self.dist_label2 = tk.Label(self, text='distance from anchor to negative:')
            self.dist_label2.pack(side=tk.BOTTOM)
            self.dist_label3 = tk.Label(self, text='distance from negative to positive:')
            self.dist_label3.pack(side=tk.BOTTOM)
        self.focus_set()
        self.load_clips()
        self.reload_display()
        self.bind('<Left>', lambda event: self.display.choice_var.set(1))
        self.bind('<Right>', lambda event: self.display.choice_var.set(2))
        self.bind('<Down>', lambda event: self.display.choice_var.set(0))
        self.bind('a', lambda event: self.display.choice_var.set(1))
        self.bind('d', lambda event: self.display.choice_var.set(2))
        self.bind('s', lambda event: self.display.choice_var.set(0))
        self.bind('e', lambda event: self.save())
        self.pack()

    def next(self, event=None):
        self.save_triplet()
        self.reload_display()

    def new_triplet(self, fps=120):
        self.i += 1
        row = self.df.iloc[self.i]
        sample_names = ['anchor', 'sample1', 'sample2']
        try:
            segments = [decode_seg_string(row[sample]) for sample in sample_names]
        except ValueError as e:
            print(e)
            # import pdb; pdb.set_trace()
            return
        self.segments = [slice(seg[0], seg[1], int(math.ceil(vid.fps/fps))) for seg in segments]
        if self.encoded is not None:
            self.clip_encodeings = [self.encoded[idx // int(len(self.video)/len(self.encoded))] for idx in random_idxs]
        self.clips = [self.video[seg] for seg in self.segments]

    def reload_display(self):
        self.display.destroy()
        self.display = ClipsDisplay(self, self.clips, fps=30)
        self.display.pack(side=tk.TOP)
        if self.encoded is not None:
            dist1 = np.linalg.norm(self.clip_encodeings[0] - self.clip_encodeings[1])
            dist2 = np.linalg.norm(self.clip_encodeings[0] - self.clip_encodeings[2])
            dist3 = np.linalg.norm(self.clip_encodeings[2] - self.clip_encodeings[1])
            print(dist1, dist2, dist3)
            self.dist_label1['text'] = f"distance between anchor and sample1: {dist1:.2f}"
            self.dist_label2['text'] = f"distance between anchor and sample2: {dist2:.2f}"
            self.dist_label3['text'] = f"distance between sample2 and sample1: {dist3:.2f}"
        self.load_clips()

    def save_triplet(self):
        self.saved_triplets.append({'video_file': self.video.video_file,
                                    'anchor': (self.segments[0].start, self.segments[0].stop),
                                    'sample1': (self.segments[1].start, self.segments[1].stop),
                                    'sample2': (self.segments[2].start, self.segments[2].stop),
                                    'selected': self.df.iloc[self.i]['selected'],
                                    'selected_verification': self.display.choice_var.get()})

    def load_clips(self):
        print("start load clips")
        # anchor, pos, neg = next(self.triplets_gen)
        # self.clips = [get_seg_clip(self.video, seg, n_frames=60, fps=120) for seg in [anchor, pos, neg]]
        self.new_triplet()
        print("finish load clips")

    def save(self):
        if not self.to_save:
            return
        print("saving dataframe...")
        df = pd.DataFrame.from_records(self.saved_triplets)
        path = 'triplets/data/selected_triplets_verificatio1.csv'
        mode = 'a' if os.path.exists(path) else 'w'
        df.to_csv(path_or_buf=path, mode=mode)
        self.saved_triplets = []

    def quit(self):
        print("quitting")
        self.save()
        self.root.quit()

data_root = Path("/home/orel/Storage/Data/K6/2020-03-26/Down")
#data_root = Path("/mnt/storage2/shuki/data/THEMIS/0015")
# landmark_file = data_root/'2020-03-23'/'Down'/'0008DeepCut_resnet50_Down2May25shuffle1_1030000.h5'
# video_file = data_root/'2020-03-23'/'Down'/'0008.MP4'


def __main__():
    print(os.getcwd())
    root = tk.Tk()
    # X_encoded_dict = np.load('models/11_03/x_encoded_dict.pkl', allow_pickle=True)
    # X_encoded_dict = {os.path.split(k)[-1][:4]: v for k, v in X_encoded_dict.items()}
    # x_encoded = X_encoded_dict[data_root.name]
    # print(x_encoded.shape, )
    video = LandmarksVideo(data_root, include_landmarks=True)
    print(video.fps)
    # print(x_encoded.shape, len(video.landmarks))
    # df = pd.read_csv('triplets/data/selected_triplets.csv')
    app = App(root, video)
    #app = VerificationApp(root, video, df.iloc[20:], encoded=x_encoded)
    root.mainloop()


if __name__ == '__main__':
    __main__()
