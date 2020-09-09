import cv2 as cv
from time import time
import numpy as np
from contextlib import contextmanager
import os
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
# import seaborn as sns

@contextmanager
def time_this(label):
    t0 = time()
    yield
    elapsed = time() - t0
    print(f"{label} took {elapsed} seconds")


def arrange_data_frame(data_frame, bodyparts_to_drop=None):
    df = deepcopy(data_frame)
    body_parts = pd.unique([i[-2] for i in df.columns])
    coords = pd.unique([i[-1] for i in df.columns])
    df.columns = pd.MultiIndex.from_product([body_parts, coords])
    df.index.name = 'index'
    df.index = df.index.map(int)
    if bodyparts_to_drop is not None:
        for part in bodyparts_to_drop:
            df.columns.levels[0].drop(part)
            df = df.applymap(float)
            df = df.drop(part, axis=1)
    xy_df = df.drop('likelihood', level=1, axis=1)
    return xy_df, body_parts

def get_rat_coords(Xs, Ys, frame_size, edge_x, edge_y):
    rat_coords = {}
    rat_coords['min_x'] = int((np.min(Xs)+np.max(Xs))/2-frame_size/2)
    rat_coords['min_y'] = int((np.min(Ys)+np.max(Ys))/2-frame_size/2)
    rat_coords['max_x'] = int((np.min(Xs)+np.max(Xs))/2+frame_size/2)
    rat_coords['max_y'] = int((np.min(Ys)+np.max(Ys))/2+frame_size/2)

    if rat_coords['min_x'] <= 0:
        rat_coords['min_x'] = 0
        rat_coords['max_x'] = frame_size
    elif rat_coords['max_x'] >= edge_x:
        rat_coords['max_x'] = edge_x - 1
        rat_coords['min_x'] = edge_x - 1 - frame_size
    if rat_coords['min_y'] <= 0:
        rat_coords['min_y'] = 0
        rat_coords['max_y'] = frame_size
    elif rat_coords['max_y'] >= edge_y:
        rat_coords['max_y'] = edge_y - 1
        rat_coords['min_y'] = edge_y - 1 - frame_size
    
    return rat_coords



def extract_frames(cap: cv.VideoCapture, start_idx, end_idx, df=None, frame_size=350):
    n_frames = end_idx - start_idx
    assert n_frames > 0
    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx)
    curr_idx = start_idx
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print("frame not found")
            break
        yield frame
    cap.release()


def extract_cut_frames(cap: cv.VideoCapture, start_idx, end_idx, df, frame_size=350):
    n_frames = end_idx - start_idx
    assert n_frames > 0
    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx)
    xy_df, body_parts = arrange_data_frame(df)
    curr_idx = start_idx
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print("frame not found")
        Xs = [xy_df[part]['x'][curr_idx] for part in body_parts]  
        Ys = [xy_df[part]['y'][curr_idx] for part in body_parts]  
        curr_idx += 1
        rat_coords = get_rat_coords(Xs, Ys, frame_size, frame.shape[1], frame.shape[0])
        bla = frame[rat_coords['min_y']:rat_coords['max_y'],rat_coords['min_x']:rat_coords['max_x'],:]
#         print(f'this frame has a size of {bla.shape[0]} over {bla.shape[1]} and the indices used to extract it were {rat_coords}')
        yield frame[rat_coords['min_y']:rat_coords['max_y'],rat_coords['min_x']:rat_coords['max_x'],:]

    cap.release()

def extract_labeled_cut_frames(cap: cv.VideoCapture, start_idx, end_idx, df, frame_size=350, bodyparts=None):

    if bodyparts is None:
        bodyparts = pd.unique([i[1] for i in df.columns])
    
    color_list = Make_color_pallete()
    n_frames = end_idx - start_idx
    assert n_frames > 0
    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx)
    xy_df, body_parts = arrange_data_frame(df)
    curr_idx = start_idx
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print("frame not found")
        else:
            # find x and y coordinates of bodyparts to cut the frame with:
            if curr_idx < 0:
                curr_idx = 1
            Xs = [xy_df[part]['x'][curr_idx] for part in body_parts]
            Ys = [xy_df[part]['y'][curr_idx] for part in body_parts]
            curr_idx += 1
            rat_coords = get_rat_coords(Xs, Ys, frame_size, frame.shape[1], frame.shape[0])
            bla = frame[rat_coords['min_y']:rat_coords['max_y'],rat_coords['min_x']:rat_coords['max_x'],:]
#         print(f'this frame has a size of {bla.shape[0]} over {bla.shape[1]} and the indices used to extract it were {rat_coords}')
            cut_frame = frame[rat_coords['min_y']:rat_coords['max_y'],rat_coords['min_x']:rat_coords['max_x'],:]
            cv.putText(cut_frame, str(start_idx), (50,50), cv.FONT_HERSHEY_SIMPLEX , 1, (255,255,255))
        # label bodyparts:
            for j, part in enumerate(body_parts):
                try:
                    x, y = int(xy_df[part]['x'][start_idx + i]), int(xy_df[part]['y'][start_idx + i])
                except:
                    orel = 1
                x, y = int(xy_df[part]['x'][start_idx+i]), int(xy_df[part]['y'][start_idx+i])
                cv.circle(cut_frame, (x-rat_coords['min_x'], y-rat_coords['min_y']), 5, color_list[j], -1)
            yield cut_frame

    cap.release()

def extract_clip(cap, start_idx, end_idx):
    n_frames = end_idx - start_idx
    assert n_frames > 0
    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx)
    frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_clip(cap: cv.VideoCapture, start_idx, end_idx, save_file, fps=30):
    write_every = int(cap.get(cv.CAP_PROP_FPS) / fps)
    writer = cv.VideoWriter()
    writer.open(filename=save_file,
                apiPreference=cv.CAP_ANY,
                fourcc=int(cap.get(cv.CAP_PROP_FOURCC)),
                fps=fps,
                frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    frame_gen = extract_frames(cap, start_idx, end_idx)
    for i, frame in enumerate(frame_gen):
        if i % write_every == 0:
            writer.write(frame)
    writer.release()
    cap.release()

def save_collage_clip(cap: cv.VideoCapture, df, frame_ids, save_file, frame_size=300, fps=30):
    write_every = int(np.ceil(cap.get(cv.CAP_PROP_FPS) / fps))
    writer = cv.VideoWriter()
    writer.open(filename=save_file,
                apiPreference=cv.CAP_ANY,
                fourcc=int(cap.get(cv.CAP_PROP_FOURCC)),
                fps=fps,
                frameSize=(frame_size*5, frame_size*3))
    # must get exactly 15 frame_ids
    frame_generators = [extract_cut_frames(cap, frame_id-100, frame_id+100, df, frame_size=frame_size) for frame_id in frame_ids]
    for i, frames in enumerate(zip(*frame_generators)):
        if i % write_every == 0:
            collage1 = np.hstack((frames[0], frames[1], frames[2], frames[3], frames[4]))
            collage2 = np.hstack((frames[5], frames[6], frames[7], frames[8], frames[9]))
            collage3 = np.hstack((frames[10],frames[11], frames[12], frames[13], frames[14]))
            frame_collage = np.vstack((collage1, collage2, collage3))
            writer.write(frame_collage)
    writer.release()
    cap.release()


def Make_color_pallete():
    color_pallete = [(0,204,0),(255,255,0),(255,102,102),(255,102,178),(102,0,204),(0,0,204),(0,128,255),(51,255,255),(204,255,204),(153,153,0),(153,0,76),(76,0,153,),(160,160,160)]
    return color_pallete

def save_clip_with_labels(cap: cv.VideoCapture, df, start_idx, end_idx, save_file, scorer=False, verbose=True, bodyparts=None, fps=60):
    
    write_every = int(np.ceil(cap.get(cv.CAP_PROP_FPS) / fps))
    writer = cv.VideoWriter()
    writer.open(filename=save_file,
                apiPreference=cv.CAP_ANY,
                fourcc=int(cap.get(cv.CAP_PROP_FOURCC)),
                fps=fps/2,
                frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    if verbose:
        print(f'clip file created')

    if scorer:
        df.columns = df.columns.droplevel(0)
        if verbose:
            print(f'scorer removed')

    if bodyparts is None:
        bodyparts = pd.unique([i[0] for i in df.columns])
        if verbose:
            print(f'bodypart names extracted')

    num_shades = len(bodyparts)
    color_list = Make_color_pallete()
    
    frame_gen = extract_frames(cap, start_idx, end_idx)
    for i, frame in enumerate(frame_gen):
        if i % write_every == 0:
            for j, part in enumerate(bodyparts):
                x, y = int(df[part]['x'][start_idx+i]), int(df[part]['y'][start_idx+i])
                cv.circle(frame, (x, y), 5, color_list[j], -1)
                # if verbose:
                    # print(f'label for {part} was added to frame {start_idx+i}')
            writer.write(frame)
            if verbose:
                print(f'frame number {start_idx+i} written')
    writer.release()
    cap.release()


def save_collage_with_labels(video_file, df, frame_ids, save_file, frame_size=350, fps=30, bodyparts=None, slowdown=2.0):
    # if you only want a specific subsets of bodyparts to be labeled, input them as a list. example -> ['nose', 'chest', 'tailBase']
    cap = cv.VideoCapture(video_file)
    write_every = int(np.ceil(cap.get(cv.CAP_PROP_FPS) / fps / slowdown))
    writer = cv.VideoWriter()
    writer.open(filename=save_file,
                apiPreference=cv.CAP_ANY,
                fourcc=int(cap.get(cv.CAP_PROP_FOURCC)),
                fps=fps,
                frameSize=(frame_size*5, frame_size*3))
    # must get exactly 15 frame_ids
    frame_generators = [extract_labeled_cut_frames(cv.VideoCapture(video_file), frame_id-100, frame_id+100, df, frame_size=frame_size, bodyparts=bodyparts) for frame_id in frame_ids]
    for i, frames in enumerate(zip(*frame_generators)):
        if i % write_every == 0:
            collage1 = np.hstack((frames[0], frames[1], frames[2], frames[3], frames[4]))
            collage2 = np.hstack((frames[5], frames[6], frames[7], frames[8], frames[9]))
            collage3 = np.hstack((frames[10], frames[11], frames[12], frames[13], frames[14]))
            frame_collage = np.vstack((collage1, collage2, collage3))
            writer.write(frame_collage)
    writer.release()
    cap.release()

def save_collage_with_labels_short(video_file, df, frame_ids, save_file, frame_size=350, fps=30, bodyparts=None,
                             slowdown=8.0, n_frames_around=20):
    # if you only want a specific subsets of bodyparts to be labeled, input them as a list. example -> ['nose', 'chest', 'tailBase']
    cap = cv.VideoCapture(video_file)
    write_every = int(np.ceil(cap.get(cv.CAP_PROP_FPS) / fps / slowdown))
    writer = cv.VideoWriter()
    writer.open(filename=save_file,
                apiPreference=cv.CAP_ANY,
                fourcc=int(cap.get(cv.CAP_PROP_FOURCC)),
                fps=fps,
                frameSize=(frame_size * 5, frame_size * 3))
    # must get exactly 15 frame_ids
    frame_generators = [extract_labeled_cut_frames(cv.VideoCapture(video_file), frame_id - n_frames_around, frame_id + n_frames_around, df,
                                                   frame_size=frame_size, bodyparts=bodyparts) for frame_id in
                        frame_ids]
    for i, frames in enumerate(zip(*frame_generators)):
        if i % write_every == 0:
            collage1 = np.hstack((frames[0], frames[1], frames[2], frames[3], frames[4]))
            collage2 = np.hstack((frames[5], frames[6], frames[7], frames[8], frames[9]))
            collage3 = np.hstack((frames[10], frames[11], frames[12], frames[13], frames[14]))
            frame_collage = np.vstack((collage1, collage2, collage3))
            writer.write(frame_collage)
    writer.release()
    cap.release()


    # fig1, ax1 = plt.subplots()
    # fig1.suptitle('Smoothed', fontsize=14)
    #
    # for idx in range(start, end):
    #     cap.set(1, idx)
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     ax1.imshow(frame)
    #
    #     for bp in bodyparts:
    #         x,y = smoothed[bp]['x'].values[idx], smoothed[bp]['y'].values[idx]
    #         ax1.scatter(x, y)
    #     fig1.show()
    #     plt.pause(0.5)
    #     ax1.cla()



if __name__ == '__main__':
    file_loc = "C:\\Data\\K6\\2020-03-30\\Down\\0014DeepCut_resnet50_DownMay7shuffle1_1030000.h5"
    df = pd.read_hdf(file_loc, index_col='scorer')
    # video_file = "data/0015DeepCut_resnet50_DownMay7shuffle1_1030000_labeled.mp4"
    video_file = "C:\\Users\\User\\OneDrive - Bar-Ilan University\\K6Data\\2020-03-30\\0014.MP4"
    save_file = "C:\\Users\\User\\OneDrive - Bar-Ilan University\\K6Data\\2020-03-30\\collage.mp4"
    cap = cv.VideoCapture(video_file)

    frame_ids = [i*1000+10000 for i in range(15)]
    
    
    # save_collage_clip(cap, df, frame_ids, save_file)
    save_collage_with_labels(cap, df, frame_ids, f"labeled_collage.mp4")
    # save_collage_with_labels(cap, df, 5500, 6500, f"clip_90.mp4", scorer=True)

    exit()
