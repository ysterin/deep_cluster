import numpy as np
import pandas as pd
from scipy import signal
from shapely.geometry import LineString
from scipy.ndimage.interpolation import shift
import logging as log
import matplotlib.pyplot as plt
import scipy.signal as sig
import cv2
from copy import deepcopy


def plot_range_N_prob(data_frame, bodypart, scope=None, axs=None):
    
    
    if axs is None:
        fig, axs = plt.subplots()

    if scope is None:
        scope = [0,-1]

    x, y = data_frame[bodypart]['x'].values, data_frame[bodypart]['y'].values
    p = data_frame[bodypart]['likelihood'].values
    scope[1] = x.shape[0]

    axs.set_title(f'x,y coordinates and likelyhood of {bodypart}')
    axs.set_xticks(np.arange(0, scope[1]-scope[0], (scope[1]-scope[0])//7))
    axs.set_xticklabels(str(np.arange(scope[0], scope[1], (scope[1]-scope[0])//7)))

    x_plot = axs.plot(x[scope[0]:scope[1]], linewidth=0.3)
    y_plot = axs.plot(y[scope[0]:scope[1]], linewidth=0.3)
    likelyhood_plot = axs.plot(100*p[scope[0]:scope[1]], linewidth=0.3)
    plt.legend([x_plot, y_plot, likelyhood_plot], ['X', 'Y', 'likelyhood'])


def get_speed(data_frame, bodypart, scope=None):

    if scope is None:
        scope = [0,-1]

    x, y = data_frame[bodypart]['x'].values[scope[0]:scope[1]], data_frame[bodypart]['y'].values[scope[0]:scope[1]]
    shift_x, shift_y = shift(x,1), shift(y,1)
    speed = np.sqrt(((np.array((x, y)).T - np.array((shift_x, shift_y)).T) ** 2).sum(1))
    return speed[1:]


def plot_speed(data_frame, bodypart, scope=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots()

    if scope is None:
        scope = [0,-1]
    
    speed = get_speed(data_frame, bodypart, scope=scope)
    scope[1] = speed.shape[0]
    fig.suptitle(f'speed (calculated by change) of {bodypart}', fontsize=14)
    axs.set_xticks(np.arange(0, scope[1]-scope[0], (scope[1]-scope[0])//7))
    axs.set_xticklabels(str(np.arange(0, scope[1]-scope[0], (scope[1]-scope[0])//7)))
    axs.plot(speed, linewidth=0.1)



def get_distance(data_frame, bp1, bp2, bp11=None, scope=None):

    if scope is None:
        scope = [0,-1]

    bp1num_x, bp1num_y = data_frame[bp1]['x'].values[scope[0]:scope[1]], data_frame[bp1]['y'].values[scope[0]:scope[1]]
    bp2num_x, bp2num_y = data_frame[bp2]['x'].values[scope[0]:scope[1]], data_frame[bp2]['y'].values[scope[0]:scope[1]]

    if bp11 is not None:
        bp11num_x, bp11num_y = data_frame[bp11]['x'].values[scope[0]:scope[1]], data_frame[bp11]['y'].values[scope[0]:scope[1]]
        bp1num_x = (bp1num_x+bp11num_x)/2
        bp1num_y = (bp1num_y+bp11num_y)/2

    distance = np.sqrt(((np.array((bp1num_x, bp1num_y)).T - np.array((bp2num_x, bp2num_y)).T) ** 2).sum(1))
    return distance


def get_angle_to_north(data_frame, bp1, bp2, scope=None):

    if scope is None:
        scope = [0,-1]

    bp1_x, bp1_y = data_frame[bp1]['x'].values[scope[0]:scope[1]], data_frame[bp1]['y'].values[scope[0]:scope[1]]
    bp2_x, bp2_y = data_frame[bp2]['x'].values[scope[0]:scope[1]], data_frame[bp2]['y'].values[scope[0]:scope[1]]

    bp2_vec = np.asarray([bp2_x-bp1_x, bp2_y-bp1_y])
    bp1_vec = np.asarray([bp1_x-bp1_x, bp1_y-bp1_y])
    north_vec = np.asarray([bp1_vec[0], bp2_vec[1]])

    angle = np.asarray([np.arccos(np.dot(bp2_vec[:,i],north_vec[:,i])/(np.linalg.norm(bp2_vec[:,i]) * np.linalg.norm(north_vec[:,i]))) for i in range(bp1_x.shape[0])])

    angle[(bp2_vec[0] > 0) & (bp2_vec[1] < 0)] = np.pi - angle[(bp2_vec[0] > 0) & (bp2_vec[1] < 0)]
    angle[(bp2_vec[0] < 0) & (bp2_vec[1] < 0)] = angle[(bp2_vec[0] < 0) & (bp2_vec[1] < 0)] + np.pi
    angle[(bp2_vec[0] < 0) & (bp2_vec[1] > 0)] = np.pi*2 - angle[(bp2_vec[0] < 0) & (bp2_vec[1] > 0)]
    return angle

def angle_3_bodyparts(data_frame, bp1, bp2, bp3, scope=None):
    # this function returns the angle between the vectors bp1-bp2 and bp2-bp3

    if scope is None:
        scope = [0,-1]
    
    bp1_x, bp1_y = data_frame[bp1]['x'].values[scope[0]:scope[1]], data_frame[bp1]['y'].values[scope[0]:scope[1]]
    bp2_x, bp2_y = data_frame[bp2]['x'].values[scope[0]:scope[1]], data_frame[bp2]['y'].values[scope[0]:scope[1]]
    bp3_x, bp3_y = data_frame[bp3]['x'].values[scope[0]:scope[1]], data_frame[bp3]['y'].values[scope[0]:scope[1]]

    vec1 = np.asarray([bp1_x-bp2_x, bp1_y-bp2_y])
    vec2 = np.asarray([bp3_x-bp2_x, bp3_y-bp2_y])

    angle = np.asarray([np.arccos(np.dot(vec1[:,i],vec2[:,i])/(np.linalg.norm(vec1[:,i]) * np.linalg.norm(vec2[:,i]))) for i in range(bp1_x.shape[0])])

    return angle

def curvature(data_frame, bp1, bp2, bp3, scope=None):

    if scope is None:
        scope = [0,-1]
    
    bp1_x, bp1_y = data_frame[bp1]['x'].values[scope[0]:scope[1]], data_frame[bp1]['y'].values[scope[0]:scope[1]]
    bp2_x, bp2_y = data_frame[bp2]['x'].values[scope[0]:scope[1]], data_frame[bp2]['y'].values[scope[0]:scope[1]]
    bp3_x, bp3_y = data_frame[bp3]['x'].values[scope[0]:scope[1]], data_frame[bp3]['y'].values[scope[0]:scope[1]]

    # Compute distance to each point
    a = np.hypot((bp1_x - bp2_x), (bp1_y - bp2_y))
    b = np.hypot((bp2_x - bp3_x), (bp2_y - bp3_y))
    c = np.hypot((bp3_x - bp1_x), (bp3_y - bp1_y))

    # Compute inverse radius of circle using surface of triangle (for which Heron's formula is used)
    k = np.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))) / 4    # Heron's formula for triangle's surface
    den = a*b*c  # Denumerator; make sure there is no division by zero.
    curvature = 4*k / den
    curvature[den==0] = 0
    return curvature

def direction(data_frame, bp, scope=None):
    
    if scope is None:
        scope = [0,-1]

    x, y = data_frame[bp]['x'].values[scope[0]:scope[1]], data_frame[bp]['y'].values[scope[0]:scope[1]]

    diff = ((np.diff(x)),(np.diff(y)))
    direc = [np.linalg.norm(d) for d in diff]
    return direc

def visualize_frame(data_frame, bodyparts, frame_idx, axs=None, vecs=None):
    # axs.plot([0,1],[0,100])
    # return
    if axs is None:
        fig, axs = plt.subplots()

    if vecs is None:
        vecs = {}
        vecs['nose2chest'] = [0, 4]
        vecs['chest2forepawR'] = [4, 2]
        vecs['chest2forepawL'] = [4, 3]
        vecs['chest2belly'] = [4, 5]
        vecs['belly2hindpawR'] = [5, 6]
        vecs['belly2hindpawR'] = [5, 7]
        vecs['belly2tailbase'] = [5, 8]
        vecs['tailbase2tail1'] = [5, 9]
        vecs['tail1totail2'] = [9, 10]
        vecs['tail22tail3'] = [10, 11]


    for bp in bodyparts:
        x, y, p = data_frame[bp]['x'][frame_idx], data_frame[bp]['y'][frame_idx], data_frame[bp]['likelihood'][frame_idx]
        axs.scatter(x, y)
        axs.annotate(bp, (x, y))

    for key in vecs:
        bp1 = bodyparts[vecs[key][0]]
        bp2 = bodyparts[vecs[key][1]]

        x1, y1 = data_frame[bp1]['x'][frame_idx], data_frame[bp1]['y'][frame_idx]
        x2, y2 = data_frame[bp2]['x'][frame_idx], data_frame[bp2]['y'][frame_idx]
        # print(f'bodypart: {bp1} is at {x1}, {y1}')
        # print(f'bodypart: {bp2} is at {x2}, {y2}')
        axs.arrow(x1,y1,x2-x1,y2-y1)

def move_direction(data_frame, bp, scope=None):
    
    if scope is None:
        scope = [0, -1]

    x, y = data_frame[bp]['x'].values[scope[0]:scope[1]], data_frame[bp]['y'].values[scope[0]:scope[1]]

    diff = ((np.diff(x)),(np.diff(y)))
    direc = [np.linalg.norm(d) for d in diff]
    return direc

def create_bsoid_features(data_frame, bodyparts=None):
    bsoid_features = {}
    bsoid_features['feat1'] = get_distance(data_frame, 'nose', 'tailbase')
    bsoid_features['feat2'] = bsoid_features['feat1'] - get_distance(data_frame, 'forepawR', 'tailbase', bp11='forePawL')
    bsoid_features['feat3'] = bsoid_features['feat1'] - get_distance(data_frame, 'hindpawR', 'tailbase', bp11='hindpawL')
    bsoid_features['feat4'] = bsoid_features['feat1'] - get_distance(data_frame, 'forepawR', 'forePawL')
    bsoid_features['feat5'] = get_speed(data_frame, 'nose')
    bsoid_features['feat6'] = get_speed(data_frame, 'tailbase')
    bsoid_features['feat7'] = get_angle_to_north(data_frame, 'nose', 'tailbase')
    return bsoid_features


def divmax_normalize(feature):
    return feature/np.max(np.abs(feature))


def normalize_to_mean(feature):
    feature -= np.mean(feature)
    return feature/ np.std(feature)

def log_divmax_normalize(feature):
    return np.log10(feature/np.max(np.abs(feature)))


def create_full_features(data_frame, bodyparts=None, norm_method = 'mean'):
    if norm_method == "mean":
        normalize = normalize_to_mean
    elif norm_method == 'divmax':
        normalize = divmax_normalize
    elif norm_method == 'log divmax':
        normalize = log_divmax_normalize
    else:
        print(f'normalization method not recodnized, using normalization to mean')
        normalize = normalize_to_mean

    full_features = {}
    # distance between nose and base of tail:
    full_features['feat1'] = normalize(get_distance(data_frame, 'nose', 'tailbase')[1:])
    # distance between forelimbs and base of tail (normalized)
    full_features['feat2'] = normalize(full_features['feat1'] - get_distance(data_frame, 'forepawR', 'tailbase', bp11='forePawL')[1:])
    # distance between hindlimbs and base of tail (normalized)
    full_features['feat3'] = normalize(full_features['feat1'] - get_distance(data_frame, 'hindpawR', 'tailbase', bp11='hindpawL')[1:])
    # distance between forelimbs (normalized)
    full_features['feat4'] = normalize(full_features['feat1'] - get_distance(data_frame, 'forepawR', 'forePawL')[1:])
    # distance between nose and chest:
    full_features['feat5'] = normalize(full_features['feat1'] - get_distance(data_frame, 'nose', 'chest')[1:])
    # speed of nose:
    full_features['feat6'] = normalize(get_speed(data_frame, 'nose'))
    # speed of tail base:
    full_features['feat7'] = normalize(get_speed(data_frame, 'tailbase'))
    # speed of forelimb right:
    full_features['feat8'] = normalize(get_speed(data_frame, 'forepawR'))
    # speed of forelimb left:
#     full_features['feat8'] = normalize(get_speed(data_frame, 'forePawL'))
    # speed of hindlimb right:
    full_features['feat9'] = normalize(get_speed(data_frame, 'forePawL'))
    # speed of hindlimb left:
    full_features['feat10'] = normalize(get_speed(data_frame, 'hindpawR'))
    # speed of forelimb right:
    full_features['feat11'] = normalize(get_speed(data_frame, 'hindpawL'))
    # cosine of the angle between nose, chest and forelimb right:
#     full_features['feat12'] = np.cos(angle_3_bodyparts(data_frame, 'nose', 'chest', 'forepawR')[1:])
#     # sine of the angle between nose, chest and forelimb right:
#     full_features['feat13'] = np.sin(angle_3_bodyparts(data_frame, 'nose', 'chest', 'forepawR')[1:])
#     # cosine of the angle between nose, chest and forelimb left:
#     full_features['feat14'] = np.cos(angle_3_bodyparts(data_frame, 'nose', 'chest', 'forePawL')[1:])
#     # sine of the angle between nose, chest and forelimb left:
#     full_features['feat15'] = np.sin(angle_3_bodyparts(data_frame, 'nose', 'chest', 'forePawL')[1:])
    #angle between tail base, belly and hindlimb right:
    full_features['feat16'] = np.cos(angle_3_bodyparts(data_frame, 'tailbase', 'hindpawR', 'belly')[1:])
    # angle between tail base, belly and hindlimb right:
    full_features['feat17'] = np.sin(angle_3_bodyparts(data_frame, 'tailbase', 'hindpawR', 'belly')[1:])
    # angle between tail base, belly and hindlimb left:
    full_features['feat18'] = np.cos(angle_3_bodyparts(data_frame, 'tailbase', 'hindpawL', 'belly')[1:])
    # angle between tail base, belly and hindlimb left:
    full_features['feat19'] = np.sin(angle_3_bodyparts(data_frame, 'tailbase', 'hindpawL', 'belly')[1:])
    # curvature between chest, belly and tail base:
    full_features['feat20'] = normalize(curvature(data_frame, 'chest', 'belly', 'tailbase')[1:])
    # curvature between nose, chest and forelimb right:
    full_features['feat21'] = normalize(curvature(data_frame, 'nose', 'chest', 'forepawR')[1:])
    # curvature between nose, chest and forelimb left:
    full_features['feat22'] = normalize(curvature(data_frame, 'nose', 'chest', 'forePawL')[1:])
    # curvature between tail base, belly and hindlimb right:
    full_features['feat23'] = normalize(curvature(data_frame, 'tailbase', 'hindpawR', 'belly')[1:])
    # curvature between tail base, belly and hindlimb left:
    full_features['feat24'] = normalize(curvature(data_frame, 'tailbase', 'hindpawL', 'belly')[1:])
    return full_features


    # distance between forearms and base of tail (normalized)
### spare code to draw vectors and debug angle finding functions:
# origin = [0], [0]
# idx = 3000
# plot, ax = plt.subplots()
# ax.arrow(0,0,bp2_vec[0,idx],bp2_vec[1,idx])
# ax.arrow(0,0,north_vec[0,idx], north_vec[1,idx])
# plt.xlim([-200, 200])
# plt.ylim([-200, 200])
# print(angle[idx]/(np.pi/180))

def __version__():
    
    return '2.2'


if __name__ == "__main__":
    pass