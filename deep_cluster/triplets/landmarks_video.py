import cv2 as cv
from PIL import Image
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
from deep_cluster.dataloader import LandmarkDataset, SequenceDataset
import os
import numpy as np
from matplotlib import cm
import torch
from torch.utils.data import ConcatDataset
import pickle
import re

data_root = Path("/home/orel/Storage/Data/K6/")


'''
a wrapper class of opencv's VideoCapture to work as acontext manager.
'''
class VideoCaptureWrapper(object):
    def __init__(self, vid_file, *args, **kwargs):
        self.vid_file = str(vid_file)
        self.vid_stream = cv.VideoCapture(self.vid_file, *args, **kwargs)

    def __enter__(self):
        self.vid_stream.open(self.vid_file)
        return self

    def __exit__(self, *args):
        self.vid_stream.release()

    def __getattr__(self, att):
        return self.vid_stream.__getattribute__(att)

    def set(self, *args):
        return self.vid_stream.set(*args)

    def get(self, *args):
        return self.vid_stream.get(*args)


'''
A general class for video files. the video can be accessed with indices or slices like an array, or work like an iterator.
'''
class Video(object):
    def __init__(self, video_file):
        self.video_file = str(video_file)
        self.cap = VideoCaptureWrapper(self.video_file)
        self.fps = int(np.round(self.cap.get(cv.CAP_PROP_FPS)))
        self.shape = np.array((self.cap.get(cv.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                              dtype=np.int)

    def __len__(self):
        return int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))

    @property
    def _shape(self):
        return np.array((self.cap.get(cv.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), dtype=np.int)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_slice(idx)
        with self.cap:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                raise Exception("frame not found")

    def get_slice(self, s):
        frames = []
        with self.cap as cap:
            step = 1 if s.step is None else s.step
            start = 0 if s.start is None else s.start
            stop = len(self) if s.stop is None else s.stop
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
            for i in range(stop - start):
                ret, frame = cap.read()
                if not ret:
                    self.cap.release()
                    raise Exception("frame not found")
                if i % step == 0:
                    frames.append(frame)
        return frames

    def __iter__(self):
        self.cap.open(self.video_file)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            self.cap.release()


def get_files(vid_dir):
    for file in os.listdir(vid_dir):
        if re.match(r'\d*\.MP4', file):
            vid_file = vid_dir / file
            vid_id = file[:4]
        elif re.match(r'\d+DeepCut.*\.h5', file):
            landmarks_file = vid_dir / file
    return vid_file, landmarks_file


def get_rotation_angle(landmarks, body_parts):
    nose_position = landmarks[body_parts.index('nose')]
    tailbase_position = landmarks[body_parts.index('tailbase')]
    vector = nose_position - tailbase_position
    angle = np.arctan2(vector[1], vector[0])
    return angle * 180 / np.pi + 90


def rotate_frames_and_landmarks(frames, landmarks, body_parts):
    h, w, *_ = frames[0].shape
    angle = get_rotation_angle(landmarks[len(frames) // 2], body_parts)
    rot_mat = cv.getRotationMatrix2D((h // 2, w // 2), angle, 0.8)
    rot_frames = [cv.warpAffine(frame, rot_mat, (h, w)) for frame in frames]
    rot_landmarks = np.append(landmarks, np.ones_like(landmarks[..., :1]), axis=-1) @ rot_mat.T
    return rot_frames, rot_landmarks


'''
A class specific to the landmark video, uses the landmarks to center around the rat and rotate to a fixed orientation, 
can also draw the landmarks on the video frames.
'''
class LandmarksVideo(object):
    color_pallete = [[int(c*255) for c in color[:3]] for color in cm.tab20.colors]

    def __init__(self, vid_dir=data_root / '2020-03-23' / 'Down', include_landmarks=True, smooth=True):
        self.vid_dir = vid_dir
        self.include_landmarks = include_landmarks
        vid_file, landmarks_file = get_files(vid_dir)
        self.vid_file = vid_file
        self.video = Video(vid_file)
        self.length = len(self.video)
        self.landmarks = LandmarkDataset(landmarks_file, normalize=False, smooth=smooth)
#         self.normalized_landmarks = LandmarkDataset(landmarks_file, normalize=True)
        self.body_parts = list(self.landmarks.body_parts)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_slice(idx)
        frame = self.video[idx]
        landmarks = self.landmarks[idx]
        min_xy, max_xy = landmarks.min(0) - 30, landmarks.max(0) + 30
        min_xy, max_xy = np.clip(min_xy, 0, self.video.shape).astype(np.int), np.clip(max_xy, 0,
                                                                                      self.video.shape).astype(np.int)
        cut_frame = frame[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]].copy()
        if self.include_landmarks:
            for j, part in enumerate(self.landmarks.body_parts):
                x, y = landmarks[j].astype(np.int) - min_xy
                cv.circle(cut_frame, (x, y), 5, self.color_pallete[j], -1)
        return cut_frame

    def get_slice(self, s):
        frames = self.video[s]
        landmarks = self.landmarks[s]
        frames, landmarks = rotate_frames_and_landmarks(frames, landmarks, self.body_parts)
        frames = np.stack(frames, axis=0)
        min_xy, max_xy = landmarks.min((0, 1)) - 30, landmarks.max((0, 1)) + 30
        min_xy, max_xy = np.clip(min_xy, 0, self.video.shape).astype(np.int), np.clip(max_xy, 0,
                                                                                      self.video.shape).astype(np.int)
        cut_frames = frames[:, min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]].copy()
        if self.include_landmarks:
            for i in range(len(landmarks)):
                for j, part in enumerate(self.landmarks.body_parts):
                    x, y = landmarks[i, j].astype(np.int) - min_xy
                    cv.circle(cut_frames[i], (x, y), 5, self.color_pallete[j], -1)
        return cut_frames

    def __getattr__(self, att):
        return self.video.__getattribute__(att)

    def set(self, *args):
        return self.video.set(*args)

    def get(self, *args):
        return self.video.get(*args)



if __name__=='__main__':
    vid_dir = data_root / '2020-03-23' / 'Down'
    print(get_files(vid_dir))