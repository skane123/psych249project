import os
import csv
from typing import Optional, Callable, Tuple, List

import av
import torch
from PIL import Image
from sklearn.datasets import get_data_home
import numpy as np
from data.base import BaseDataset
from data.augmentations import rand_augment_transform, _FILL, randaugment_parallel, randaugment_threaded
import shutil


class SSV2Pruned(BaseDataset):
    """Dataset for the pruned SomethingSomethingV2 videos, using fetch_and_extract."""

    BUCKET = "gs://bbscore_datasets/SSV2_pruned"

    # Preset list of original class IDs (sparse)
    ORIGINAL_CLASSES = [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 12,
        13, 14, 15, 16, 22, 23, 31, 33, 38,
        39, 46, 51, 53, 90, 91, 92, 122,
        124, 130, 138, 139, 141, 143, 144,
        150, 151, 156, 170, 171, 173
    ]

    # Mapping from original class ID to dense 0..N-1 index
    CLASS_TO_IDX = {orig: idx for idx, orig in enumerate(ORIGINAL_CLASSES)}

    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None,
        apply_randaugment: bool = False,
        randaugment_config_str: str = 'rand-m7-n2-mstd0.3',
        train: bool = True,
    ):
        super().__init__(root_dir)
        self.overwrite = overwrite
        self.preprocess = preprocess
        self.train = train
        self.apply_randaugment = apply_randaugment
        self.randaugment_transform = None

        if self.apply_randaugment and self.train:
            hparams_vid = {"img_mean": _FILL,
                           "translate_const": 50, "magnitude_std": 0.3}
            self.randaugment_transform = rand_augment_transform(
                randaugment_config_str, hparams_vid)

        self.stimulus_data: List[str] = []
        self.labels: List[int] = []
        self.target_fps = 12.0
        self.video_folder_name = (
            "modified_train_videos/unmodified"
            if train
            else "modified_test_videos/unmodified"
        )

        self.video_path = os.path.join(self.root_dir, self.video_folder_name)
        self._prepare_videos()

    def _download_ssv2_data(self):
        """No-op because data is already locally preprocessed."""
        pass

    def _prepare_videos(self):
        """Load label mapping from CSV and build the sorted list of video paths."""
        self._download_ssv2_data()

        csv_path = os.path.join(
            self.root_dir,
            "modified_train_videos/sample_train_classes_map.csv" if self.train else "modified_test_videos/sample_test_classes_map.csv"
        )

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Loading labels from {csv_path}...")

        with open(csv_path, newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                full_path = row["filename"]
                orig_cls = int(row["class"])

                # Check if the original class is in our mapping
                if orig_cls not in self.CLASS_TO_IDX:
                    print(
                        f"WARNING: Class {orig_cls} not found in CLASS_TO_IDX mapping")
                    continue

                video_file_path = os.path.join(self.video_path, full_path + ".npy")

                # Check if video file actually exists
                if not os.path.isfile(video_file_path):
                    print(f"WARNING: Video file not found: {video_file_path}")
                    continue

                self.stimulus_data.append(video_file_path)
                # Remap to dense index
                self.labels.append(self.CLASS_TO_IDX[orig_cls])

        print(f"Loaded {len(self.stimulus_data)} video samples")

        if len(self.stimulus_data) == 0:
            raise ValueError("No valid video samples found!")

    def __len__(self) -> int:
        return len(self.stimulus_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.stimulus_data[idx]
        frames = self._load_data(video_path)
        if self.randaugment_transform and self.train:
            frames = randaugment_threaded(
                frames, self.randaugment_transform, num_workers=8)
        label = self.labels[idx]
        return self.preprocess(frames), label

    def _load_data(self, video_path: str) -> List[Image.Image]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Frame file not found: {video_path}")

        try:
            arr = np.load(video_path)  # (T, H, W, 3)

            frames = [Image.fromarray(frame) for frame in arr]

            return frames

        except Exception as e:
            print(f"Error loading frames {video_path}: {e}")
            raise


class SSV2PrunedStimulusTrainSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join("/content/drive/MyDrive/Psych249", SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite,
                         preprocess=preprocess, train=True)


class AugmentedSSV2PrunedStimulusTrainSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join("/content/drive/MyDrive/Psych249", SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite, preprocess=preprocess, train=True,
                         apply_randaugment=True)


class SSV2PrunedStimulusTestSet(SSV2Pruned):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        overwrite: bool = False,
        preprocess: Optional[Callable] = None
    ):
        if root_dir is None:
            root_dir = os.path.join("/content/drive/MyDrive/Psych249", SSV2Pruned.__name__)
        super().__init__(root_dir=root_dir, overwrite=overwrite,
                         preprocess=preprocess, train=False)
