import random
import pandas as pd
import numpy as np
from typing import Optional, Callable, List
import os
import av
import shutil
from PIL import Image
from tqdm import tqdm
from itertools import islice

from sklearn.datasets import get_data_home

# from SSV2_pruned import SSV2Pruned

random.seed(10)

def load_data(video_path: str, target_fps: int = 12) -> List[Image.Image]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        orig_fps = float(stream.average_rate)
        duration_s = (container.duration or 0) / 1e6
        total_frames = int(orig_fps * duration_s)

        interval = max(int(round(orig_fps / target_fps)), 1)
        idxs = list(range(0, total_frames, interval))

        max_count = int(target_fps * 4)
        if len(idxs) > max_count:
            idxs = idxs[:max_count]

        frames: List[Image.Image] = []
        frame_idx = 0
        next_ptr = 0

        for packet in container.demux(stream):
            for frm in packet.decode():
                if next_ptr < len(idxs) and frame_idx == idxs[next_ptr]:
                    frames.append(frm.to_image())
                    next_ptr += 1
                    if next_ptr >= len(idxs):
                        break
                frame_idx += 1
            if next_ptr >= len(idxs):
                break

        container.close()

        if frames and len(frames) < max_count:
            frames += [frames[-1]] * (max_count - len(frames))

        return frames

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        raise


def save_video(frames, output_path, fps=12):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    container = av.open(output_path, mode="w")

    # WebM-compatible codec
    stream = container.add_stream("vp9", rate=fps)

    width, height = frames[0].size
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        frame_array = np.array(frame)
        video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")

        for packet in stream.encode(video_frame):
            container.mux(packet)

    # flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()

def save_frames(frames, output_path):
    """
    Save frames as a numpy array for fast future loading.

    frames: List[PIL.Image]
    output_path: path ending in .npy
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames_arr = np.stack([np.asarray(f) for f in frames])  # (T, H, W, 3)

    np.save(output_path, frames_arr, allow_pickle=False)

def load_frames(path):
    arr = np.load(path)

    frames = [Image.fromarray(frame) for frame in arr]

    return frames



def randomize_frames(frames: List[Image.Image]) -> List[Image.Image]:
    randomized_frames = random.sample(frames, len(frames))
    return randomized_frames

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def local_randomize_frames(frames: List[Image.Image]) -> List[Image.Image]:
    chunk_size = 6
    chunks = list(batched(frames, chunk_size))
    randomized_chunks = [random.sample(list(chunk), len(chunk)) for chunk in chunks]
    return [frame for chunk in randomized_chunks for frame in chunk]

def gap_frames(frames: List[Image.Image]) -> List[Image.Image]:
    chunk_size = 3
    chunks = list(batched(frames, chunk_size))

    gapped_chunks = chunks[::2]
    return [frame for chunk in gapped_chunks for frame in chunk]



### MAIN CODE

data_home = get_data_home()
ssv2_data_loc = data_home + "/SSV2Pruned"


test_vid_dir = os.path.join(ssv2_data_loc, "modified_test_videos")
os.makedirs(test_vid_dir, exist_ok=True)

train_vid_dir = os.path.join(ssv2_data_loc, "modified_train_videos")
os.makedirs(train_vid_dir, exist_ok=True)


test_map_path = ssv2_data_loc + "/test_classes_map.csv"
test_map = pd.read_csv(test_map_path)

train_map_path = ssv2_data_loc + "/train_classes_map.csv"
train_map = pd.read_csv(train_map_path)


test_sample_size = 1000
test_sample_map = test_map.sample(n=test_sample_size, random_state=10)
test_sample_map['video_path'] = ssv2_data_loc + '/ssv2_test_videos/' + test_sample_map['filename']
test_sample_map.to_csv(test_vid_dir + '/sample_test_classes_map.csv')

train_sample_size = 10000
train_sample_map = train_map.sample(n=train_sample_size, random_state=10)
train_sample_map['video_path'] = ssv2_data_loc + '/ssv2_train_videos/' + train_sample_map['filename']
train_sample_map.to_csv(train_vid_dir + '/sample_train_classes_map.csv')


test_unmodified_vid_dir = os.path.join(test_vid_dir, "unmodified")
os.makedirs(test_unmodified_vid_dir, exist_ok=True)

test_randomized_vid_dir = os.path.join(test_vid_dir, "randomized")
os.makedirs(test_randomized_vid_dir, exist_ok=True)

test_local_randomized_vid_dir = os.path.join(test_vid_dir, "local_randomized")
os.makedirs(test_local_randomized_vid_dir, exist_ok=True)

test_gapped_vid_dir = os.path.join(test_vid_dir, "gapped")
os.makedirs(test_gapped_vid_dir, exist_ok=True)


train_unmodified_vid_dir = os.path.join(train_vid_dir, "unmodified")
os.makedirs(train_unmodified_vid_dir, exist_ok=True)

train_randomized_vid_dir = os.path.join(train_vid_dir, "randomized")
os.makedirs(train_randomized_vid_dir, exist_ok=True)

train_local_randomized_vid_dir = os.path.join(train_vid_dir, "local_randomized")
os.makedirs(train_local_randomized_vid_dir, exist_ok=True)

train_gapped_vid_dir = os.path.join(train_vid_dir, "gapped")
os.makedirs(train_gapped_vid_dir, exist_ok=True)


fps = 12

print('Processing test data')
for path in tqdm(test_sample_map['video_path']):
    unmodified_frames = load_data(path, target_fps=fps)
    unmodified_path = os.path.join(test_unmodified_vid_dir, os.path.basename(path))
    save_frames(unmodified_frames, unmodified_path)
    # unmodified_frames = load_frames(unmodified_path + ".npy")
    # save_video(unmodified_frames, unmodified_path, fps=fps)

    randomized_frames = randomize_frames(unmodified_frames)
    randomized_path = os.path.join(test_randomized_vid_dir, os.path.basename(path))
    save_frames(randomized_frames, randomized_path)
    # randomized_frames = load_frames(randomized_path + ".npy")
    # save_video(randomized_frames, randomized_path, fps=fps)

    local_randomized_frames = local_randomize_frames(unmodified_frames)
    local_randomized_path = os.path.join(test_local_randomized_vid_dir, os.path.basename(path))
    save_frames(local_randomized_frames, local_randomized_path)
    # local_randomized_frames = load_frames(local_randomized_path + ".npy")
    # save_video(local_randomized_frames, local_randomized_path, fps=fps)

    gapped_frames = gap_frames(unmodified_frames)
    gapped_path = os.path.join(test_gapped_vid_dir, os.path.basename(path))
    save_frames(gapped_frames, gapped_path)
    # gapped_frames = load_frames(gapped_path + ".npy")
    # save_video(gapped_frames, gapped_path, fps=fps)

print('Processing train data')
for path in tqdm(train_sample_map['video_path']):
    unmodified_frames = load_data(path, target_fps=fps)
    unmodified_path = os.path.join(train_unmodified_vid_dir, os.path.basename(path))
    save_frames(unmodified_frames, unmodified_path)
    # unmodified_frames = load_frames(unmodified_path + ".npy")
    # save_video(unmodified_frames, unmodified_path, fps=fps)

    randomized_frames = randomize_frames(unmodified_frames)
    randomized_path = os.path.join(train_randomized_vid_dir, os.path.basename(path))
    save_frames(randomized_frames, randomized_path)
    # randomized_frames = load_frames(randomized_path + ".npy")
    # save_video(randomized_frames, randomized_path, fps=fps)

    local_randomized_frames = local_randomize_frames(unmodified_frames)
    local_randomized_path = os.path.join(train_local_randomized_vid_dir, os.path.basename(path))
    save_frames(local_randomized_frames, local_randomized_path)
    # local_randomized_frames = load_frames(local_randomized_path + ".npy")
    # save_video(local_randomized_frames, local_randomized_path, fps=fps)

    gapped_frames = gap_frames(unmodified_frames)
    gapped_path = os.path.join(train_gapped_vid_dir, os.path.basename(path))
    save_frames(gapped_frames, gapped_path)
    # gapped_frames = load_frames(gapped_path + ".npy")
    # save_video(gapped_frames, gapped_path, fps=fps)





