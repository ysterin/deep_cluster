import numpy as np
from matplotlib import pyplot as plt
import typing
from pathlib import Path
import pickle
from dataclasses import dataclass, field


class Segment:
    def __init__(self, cluster_id, start_frame, n_frames, x_encoded):
        self.cluster_id = cluster_id
        self.start_frame = start_frame
        self.n_frames = n_frames
        self.encoded_mean = x_encoded[self.start_frame // 4: self.start_frame // 4 + self.n_frames // 4].mean(axis=0)


def load_segments(model_dir, landmark_file):
    model_dir = Path(model_dir)
    with open(model_dir / 'labels_dict.pkl', 'rb') as file:
        labels_dict = pickle.load(file)

    with open(model_dir / 'data_dict.pkl', 'rb') as file:
        data_dict = pickle.load(file)

    with open(model_dir / 'segments_dict.pkl', 'rb') as file:
        segments_dict = pickle.load(file)

    with open(model_dir / 'x_encoded_dict.pkl', 'rb') as file:
        x_encoded_dict = pickle.load(file)

    segments = segments_dict[landmark_file]
    labels = labels_dict[landmark_file]
    x_encoded = x_encoded_dict[landmark_file]
    labels = labels_dict[landmark_file]
    n_clusters = max(labels) + 1
    segments = [Segment(*seg, x_encoded) for seg in segments]
    return segments


def get_pos_neg(anchor_id, encoded, labels, min_dist=9.0, max_dist=10.0):
    anchor = encoded[anchor_id]
    dists = np.linalg.norm(encoded - anchor, axis=1)
    in_dist = np.where(np.logical_and(min_dist < dists, dists < max_dist))[0]
    positives = in_dist[labels[in_dist] == labels[anchor_id]]
    negatives = in_dist[labels[in_dist] != labels[anchor_id]]
    pos_id = np.random.choice(positives)
    neg_id = np.random.choice(negatives)
    return pos_id, neg_id


def sample_triplet(encoded, labels):
    anchor_id = np.random.choice(len(encoded))
    anchor = encoded[anchor_id]
    dists = np.linalg.norm(encoded - anchor, axis=-1)
    nums, bins, *_ = plt.hist(dists, bins=100)
    plt.close()
    to_id = np.where(np.cumsum(nums) / sum(nums) > 0.9)[0][0]
    bins = bins[:to_id]
    bin_idxs = [np.where
        (np.logical_and(bins[:-1, np.newaxis] < dists[np.newaxis], dists[np.newaxis] < bins[1:, np.newaxis])[i])[0] for i in range(len(bins ) -1)]
    n_same_cluster_in_bin = np.array([(labels[bin_idxs[i]] == labels[anchor_id]).sum() for i in range(len(bins) - 1)])
    n_other_clusters_in_bin = np.array([(labels[bin_idxs[i]] != labels[anchor_id]).sum() for i in range(len(bins) - 1)])
    ratio = n_same_cluster_in_bin / (n_same_cluster_in_bin + n_other_clusters_in_bin + 1e-6)
    try:
        min_bin_idx = np.where(ratio > 0.5)[0][-1]
    except IndexError:
        min_bin_idx = 0
    max_bin_idx = np.where(np.logical_and(ratio < 0.5, np.arange(len(bins) - 1) >= min_bin_idx))[0][0]
    min_dist, max_dist = bins[min_bin_idx], bins[max_bin_idx +1]
    assert min_dist <= max_dist, (min_dist, max_dist)
    assert np.sum(nums[min_bin_idx: max_bin_idx +1]) > 0
    print(ratio[min_bin_idx:max_bin_idx + 1].mean())
    in_dist = np.where(np.logical_and(min_dist < dists, dists < max_dist))[0]
    if len(in_dist) < 5:
        raise Exception('not enough samples in distance range')
    positives = in_dist[labels[in_dist] == labels[anchor_id]]
    negatives = in_dist[labels[in_dist] != labels[anchor_id]]
    pos_id = np.random.choice(positives)
    neg_id = np.random.choice(negatives)
    return anchor_id, pos_id, neg_id


def triplets_segments_gen(segments, n_triplets=100):
    segment_encs = np.stack([seg.encoded_mean for seg in segments])
    segment_cluster_ids = np.array([seg.cluster_id for seg in segments])
    i = 0
    while True:
        try:
            anchor_id, pos_id, neg_id = sample_triplet(encoded=segment_encs, labels=segment_cluster_ids)
            anchor, pos, neg = segments[anchor_id], segments[pos_id], segments[neg_id]
            if min([seg.n_frames for seg in [anchor, pos, neg]]) <= 20:
                continue
            if max([seg.n_frames for seg in [anchor, pos, neg]]) > 480:
                continue
            if max([seg.n_frames for seg in [anchor, pos, neg]]) / min \
                    ([seg.n_frames for seg in [anchor, pos, neg]]) > 3:
                continue
            yield anchor, pos, neg
        except Exception:
            continue
        i += 1
        if i == n_triplets:
            break


if __name__ == '__main__':
    data_root = Path("/home/orel/Storage/Data/K6/")
    landmark_file = data_root / '2020-03-23' / 'Down' / '0008DeepCut_resnet50_Down2May25shuffle1_1030000.h5'
    segments = load_segments(model_dir='../models/11_03', landmark_file=landmark_file)
    for tri in triplets_segments_gen(segments, 3):
        print(tri)

