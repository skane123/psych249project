import os
import pickle
import configparser
import datetime
import getpass
import numpy as np
import subprocess  # For git commands
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import get_data_home
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
from collections import Counter

from data.utils import custom_collate
from extractor_wrapper_online import OnlineFeatureExtractor, IdentityFeatureExtractor
from metrics import METRICS  # Access to all metrics including online ones
from models import get_model_class_and_id, MODEL_REGISTRY
from tqdm import tqdm
from torch.utils.data import random_split, Subset

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.backends\.cuda\.sdp_kernel\(\).*deprecated.*",
    category=FutureWarning,
)


def _run_git(cmd):
    try:
        out = subprocess.check_output(["git"] + cmd, stderr=subprocess.DEVNULL)
        return out.strip().decode()
    except Exception:
        return None


def get_local_commit(): return _run_git(["rev-parse", "HEAD"])
def is_worktree_clean(): return _run_git(["status", "--porcelain"]) == ""


def get_upstream_commit():
    upstream = _run_git(["rev-parse", "--abbrev-ref",
                        "--symbolic-full-name", "@{u}"])
    return _run_git(["rev-parse", upstream]) if upstream else None


def can_stratify(labels, min_samples_per_class=2):
    """
    Check if stratification is possible with given labels

    Args:
        labels: List or array of labels
        min_samples_per_class: Minimum samples needed per class for stratification

    Returns:
        bool: True if stratification is viable, False otherwise
    """
    try:
        # Check if labels are discrete (not continuous floats)
        labels_array = np.array(labels)

        # If labels are float vectors/arrays, can't stratify
        if labels_array.dtype == np.float32 or labels_array.dtype == np.float64:
            if labels_array.ndim > 1:  # Multi-dimensional vectors
                return False
            # For 1D floats, check if they're actually discrete values
            unique_vals = np.unique(labels_array)
            if len(unique_vals) > len(labels_array) * 0.5:  # Too many unique values
                return False

        # Check class distribution
        label_counts = Counter(labels)
        min_count = min(label_counts.values())

        # Need at least min_samples_per_class samples per class
        if min_count < min_samples_per_class:
            return False

        # Need at least 2 different classes
        if len(label_counts) < 2:
            return False

        return True

    except Exception:
        return False


def robust_train_val_split(dataset, test_size=0.1, random_state=42,
                           get_label_fn=None, verbose=True):
    """
    Automatically choose between stratified and random splitting based on label characteristics.
    Optimized to check for dataset.labels first before iterating through the dataset.

    Args:
        dataset: PyTorch dataset
        test_size: Fraction for validation set
        random_state: Random seed for reproducibility
        get_label_fn: Function to extract label from dataset item. If None, assumes label at index 1
        verbose: Whether to print splitting strategy used

    Returns:
        train_indices, val_indices: Arrays of indices for train and validation sets
    """
    # Extract labels efficiently
    if get_label_fn is None:
        def get_label_fn(item): return item[1]  # Default: label at index 1

    labels = None

    # First, check if dataset has a labels attribute
    if hasattr(dataset, 'labels') and dataset.labels is not None:
        if verbose:
            print("Found dataset.labels attribute, using it directly")
        labels = dataset.labels

        # Convert to list if it's not already
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        elif not isinstance(labels, (list, tuple)):
            labels = list(labels)

    # If no labels attribute found, check for targets attribute (common in torchvision datasets)
    elif hasattr(dataset, 'targets') and dataset.targets is not None:
        if verbose:
            print("Found dataset.targets attribute, using it directly")
        labels = dataset.targets

        # Convert to list if it's not already
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        elif not isinstance(labels, (list, tuple)):
            labels = list(labels)

    elif hasattr(dataset, 'neural_data') and dataset.neural_data is not None:
        if verbose:
            print("Found dataset.neural_data attribute, using it directly")
        labels = dataset.neural_data

        # Convert to list if it's not already
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        elif not isinstance(labels, (list, tuple)):
            labels = list(labels)

    elif hasattr(dataset, '_torchvision_dataset') and dataset._torchvision_dataset.targets is not None:
        if verbose:
            print("Found dataset._torchvision_dataset.targets, using it directly")
        labels = dataset._torchvision_dataset.targets

        # Convert to list if it's not already
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        elif not isinstance(labels, (list, tuple)):
            labels = list(labels)

    # If still no labels found, fall back to iterating through the dataset
    if labels is None:
        if verbose:
            print(
                "No labels/targets attribute found, extracting labels by iterating through dataset")

        labels = []
        for i in tqdm(range(len(dataset))):
            try:
                label = get_label_fn(dataset[i])
                labels.append(label)
            except Exception as e:
                if verbose:
                    print(
                        f"Warning: Could not extract label for sample {i}: {e}")
                    print("Falling back to random split")
                return train_test_split(
                    np.arange(len(dataset)),
                    test_size=test_size,
                    random_state=random_state
                )

    indices = np.arange(len(dataset))

    # Check if we can stratify
    if can_stratify(labels):
        try:
            if verbose:
                print("Using stratified split to maintain label distribution")

            train_indices, val_indices = train_test_split(
                indices,
                test_size=test_size,
                stratify=labels,
                random_state=random_state
            )

            # Verify the split worked
            if verbose:
                train_label_dist = Counter([labels[i] for i in train_indices])
                val_label_dist = Counter([labels[i] for i in val_indices])
                print(f"Train label distribution: {dict(train_label_dist)}")
                print(f"Val label distribution: {dict(val_label_dist)}")

            return train_indices, val_indices

        except Exception as e:
            if verbose:
                print(f"Stratification failed: {e}")
                print("Falling back to random split")
    else:
        if verbose:
            print("Cannot stratify (continuous labels or insufficient samples per class)")
            print("Using random split")

    # Fall back to random split
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state
    )


class OnlineBenchmarkScore:
    def __init__(
        self,
        stimulus_train_class,  # Class for training stimuli
        model_identifier: str,
        layer_name: str,
        num_classes: int,  # For classifier metrics, 0 or 1 for regression
        stimulus_test_class,  # Optional
        # Assembly related args are less relevant for online behavioral metrics,
        # but kept for structural similarity if some online neural metrics arise.
        assembly_class=None,
        assembly_train_kwargs=None,
        assembly_test_kwargs=None,
        # Batch size for dataloaders, not for feature extractor directly for online
        dataloader_batch_size: int = 32,
        num_workers: int = 4,
        sequence_mode_for_extractor: str = "all",  # "last", "all", "concatenate"
        debug: bool = False,
        val_split: float = 0.1,  # Validation split ratio
        random_state: int = 42,  # Random seed for reproducibility
        # Custom label extraction function
        get_label_fn: Optional[callable] = None,
        verbose: bool = True  # Whether to print splitting info
    ):
        self.debug = debug
        self.model_identifier = model_identifier
        self.layer_name = layer_name
        self.num_classes = num_classes  # Crucial for online classifiers
        self.sequence_mode_for_extractor = sequence_mode_for_extractor
        self.val_split = val_split
        self.random_state = random_state
        self.get_label_fn = get_label_fn
        self.verbose = verbose

        # Instantiate model and preprocessing function provider
        self.model_class_provider, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance_provider = self.model_class_provider()  # This is e.g., VideoMAE()

        # Get the actual nn.Module model
        self.pytorch_model = self.model_instance_provider.get_model(
            self.model_id_mapping)

        # Instantiate OnlineFeatureExtractor
        self.extractor = OnlineFeatureExtractor(
            model=self.pytorch_model,
            layer_name=self.layer_name,
            postprocess_fn=self.model_instance_provider.postprocess_fn,
            static=self.model_instance_provider.static,
            sequence_mode=self.sequence_mode_for_extractor
        )

        # Prepare Stimuli DataLoaders with robust splitting
        self._setup_dataloaders(
            stimulus_train_class,
            stimulus_test_class,
            dataloader_batch_size,
            num_workers
        )

        # Get dummy to cause error if fallback is necessary
        dummy_batch_data, _ = next(iter(self.train_dataloader))

        self.online_metrics_to_run: Dict[str, OnlineMetric] = {}

        # Data home, results path (similar to BenchmarkScore)
        default_data_home = "/content/drive/MyDrive/Psych249"
        # if RESULTS_PATH is set, use it, otherwise use default_data_home
        results_base = os.environ.get('RESULTS_PATH', default_data_home)

        self.data_home = default_data_home
        self.benchmark_name = self.__class__.__name__

        self.results_file = os.path.join(
            results_base, 'results',
            f"online_{self.model_identifier}_{layer_name}_{self.benchmark_name}.pkl"
        )
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        try:
            if self.verbose:
                print(
                    "Attempting to infer feature dimensionality for internal models...")
            # Get one batch from train_dataloader
            dummy_batch_data, _ = next(iter(self.train_dataloader))

            # If dummy_batch_data is a tuple/list (e.g. (stim_ids, tensor_data)), get tensor part
            if isinstance(dummy_batch_data, (list, tuple)) and isinstance(dummy_batch_data[0], list) and isinstance(dummy_batch_data[1], torch.Tensor):
                dummy_batch_data = dummy_batch_data[1]
                self.input_feature_dim_for_metric = self.extractor.get_feature_dimensionality(
                    dummy_batch_data.to(self.extractor.device))
            elif isinstance(dummy_batch_data, torch.Tensor):
                self.input_feature_dim_for_metric = self.extractor.get_feature_dimensionality(
                    dummy_batch_data.to(self.extractor.device))
            elif isinstance(dummy_batch_data, dict):
                dummy_batch_data = dummy_batch_data
                self.input_feature_dim_for_metric = self.extractor.get_feature_dimensionality(
                    dummy_batch_data)
            elif not isinstance(dummy_batch_data, torch.Tensor):
                raise ValueError(
                    f"Could not get a tensor from dummy_batch_data, type: {type(dummy_batch_data)}")

            if self.verbose:
                print(
                    f"Inferred input_feature_dim for metric: {self.input_feature_dim_for_metric}")
        except Exception as e:
            raise ValueError(f"Could not automatically infer input_feature_dim: {e}. "
                             "Ensure the OnlineMetric receives this, or override in subclass.")

    def _setup_dataloaders(self, stimulus_train_class, stimulus_test_class,
                           dataloader_batch_size, num_workers):
        """Setup train, validation, and test dataloaders with robust splitting"""

        # Handle tuple input for augmented/non-augmented datasets
        stimulus_train_noaugment_class = None
        if isinstance(stimulus_train_class, tuple):
            stimulus_train_class, stimulus_train_noaugment_class = stimulus_train_class

        if isinstance(dataloader_batch_size, list):
            dataloader_batch_size = dataloader_batch_size[0]
        # Instantiate the full training dataset
        full_train_ds = stimulus_train_class(
            preprocess=self.model_instance_provider.preprocess_fn
        )
        if stimulus_train_noaugment_class:
            full_train_noaugment_ds = stimulus_train_noaugment_class(
                preprocess=self.model_instance_provider.preprocess_fn
            )

        # Perform robust train/val split
        train_indices, val_indices = robust_train_val_split(
            full_train_ds,
            test_size=self.val_split,
            random_state=self.random_state,
            get_label_fn=self.get_label_fn,
            verbose=self.verbose
        )

        # Create subsets
        train_ds = Subset(full_train_ds, train_indices)
        if stimulus_train_noaugment_class:
            val_ds = Subset(full_train_noaugment_ds, val_indices)
        else:
            val_ds = Subset(full_train_ds, val_indices)

        if self.verbose:
            print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        # Create train and validation dataloaders
        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=dataloader_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate,
            drop_last=True,
        )

        self.val_dataloader = DataLoader(
            val_ds,
            batch_size=dataloader_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate,
        )

        # Create test dataloader
        self.stimulus_test_dataset = stimulus_test_class(
            preprocess=self.model_instance_provider.preprocess_fn
        )
        self.test_dataloader = DataLoader(
            self.stimulus_test_dataset,
            batch_size=dataloader_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate
        )

    def add_metric(self, metric_name: str, metric_params: Optional[Dict] = None):
        if metric_name not in METRICS:
            raise ValueError(
                f"Metric '{metric_name}' not found in METRICS registry.")
        metric_class = METRICS[metric_name]

        # Check if it's a subclass of OnlineMetric (conceptual check, actual check might be `isinstance`)
        # For now, we assume if it's in a certain naming convention or if the user knows.

        # Default params for the metric, can be overridden by metric_params
        # Crucially, pass num_classes and the inferred input_feature_dim
        default_metric_init_params = {
            "num_classes": self.num_classes,
            "input_feature_dim": self.input_feature_dim_for_metric,
        }
        if metric_params:
            default_metric_init_params.update(metric_params)

        try:
            self.online_metrics_to_run[metric_name] = metric_class(
                **default_metric_init_params)
        except TypeError as e:
            raise ValueError(f"Error initializing metric '{metric_name}' with params {default_metric_init_params}. "
                             f"Original error: {e}. Check if the metric class is an OnlineMetric and "
                             f"if input_feature_dim was correctly inferred or needs to be provided via metric_params.")

    def run(self) -> Dict:
        all_metric_results = {}

        for metric_name, metric_instance in self.online_metrics_to_run.items():
            print(f"\n--- Running Online Metric: {metric_name} ---")

            # The metric's compute method handles the training and evaluation loop
            # It uses the extractor and dataloaders passed here.
            metric_results = metric_instance.compute(
                extractor=self.extractor,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,  # Pass val_dataloader if available
                test_dataloader=self.test_dataloader  # Pass test_dataloader for final eval
            )
            all_metric_results[metric_name] = metric_results

        # Save all results to a local pickle file
        # The structure of the pickle file will be a dict keyed by metric_name
        final_output_to_pickle = {
            "metrics": all_metric_results,
            "ceiling": None,  # Placeholder, online metrics handle their own scores mostly
            "model_identifier": self.model_identifier,
            "layer_name": self.layer_name,
            "benchmark_name": self.benchmark_name
        }

        with open(self.results_file, 'wb') as f:
            pickle.dump(final_output_to_pickle, f)
        print(f"\nSaved all online metric results to {self.results_file}")

        return final_output_to_pickle


class OnlineAssemblyBenchmarkScorer:
    def __init__(
        self,
        source_assembly_class,
        target_assembly_class,
        source_assembly_train_kwargs=None,
        source_assembly_test_kwargs=None,
        target_assembly_train_kwargs=None,
        target_assembly_test_kwargs=None,
        batch_size=256,
        metrics=None,
        debug=False
    ):
        self.source_class = source_assembly_class
        self.target_class = target_assembly_class
        self.src_train_kwargs = source_assembly_train_kwargs or {}
        self.src_test_kwargs = source_assembly_test_kwargs or {}
        self.tgt_train_kwargs = target_assembly_train_kwargs or {}
        self.tgt_test_kwargs = target_assembly_test_kwargs or {}

        self.batch_size = batch_size
        self.debug = debug
        self.metrics = metrics or []
        self.metric_params = {}

        # Setup Results Directory
        data_home = "/content/drive/MyDrive/Psych249"
        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def add_metric(self, metric_name, metric_params=None):
        """Adds a metric to the list of metrics to run."""
        if metric_name not in self.metrics:
            self.metrics.append(metric_name)
        if metric_params:
            self.metric_params[metric_name] = metric_params

    def _prepare_dataloader(self, source_data, target_data, shuffle=False):
        src_t = torch.tensor(source_data, dtype=torch.float32)
        tgt_t = torch.tensor(target_data, dtype=torch.float32)
        dataset = TensorDataset(src_t, tgt_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def run(self):
        # 1. Load Data (Offline style)
        print("Loading Source Assembly...")
        src_loader = self.source_class()
        src_train, _ = src_loader.get_assembly(**self.src_train_kwargs)
        src_test, _ = src_loader.get_assembly(**self.src_test_kwargs)

        print("Loading Target Assembly...")
        tgt_loader = self.target_class()
        tgt_train, ceiling = tgt_loader.get_assembly(**self.tgt_train_kwargs)
        tgt_test, _ = tgt_loader.get_assembly(**self.tgt_test_kwargs)

        print(f"Data Loaded. Train: {src_train.shape} -> {tgt_train.shape}")

        # 2. Split Train into Train/Val for Online Metric (90/10 split)
        n_total = len(src_train)
        n_train = int(0.9 * n_total)
        # Fix seed for reproducibility of split
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        # 3. Create DataLoaders
        train_loader = self._prepare_dataloader(
            src_train[train_idx], tgt_train[train_idx], shuffle=True)
        val_loader = self._prepare_dataloader(
            src_train[val_idx], tgt_train[val_idx], shuffle=False)
        test_loader = self._prepare_dataloader(
            src_test, tgt_test, shuffle=False)

        # 4. Setup Fake Extractor
        extractor = IdentityFeatureExtractor()

        # 5. Run Metrics
        results = {}
        # Determine dimensions for metric initialization
        # Check if data is (B, T, C) or (B, C)
        if src_train.ndim == 3:
            input_dim = src_train.shape[2]
        else:
            input_dim = src_train.shape[1]

        output_dim = tgt_train.shape[1]

        for metric_name in self.metrics:
            print(f"--- Running Metric: {metric_name} ---")
            if metric_name not in METRICS:
                print(
                    f"Warning: Metric '{metric_name}' not found in registry. Skipping.")
                continue

            metric_cls = METRICS[metric_name]

            # Initialize Metric
            init_params = {
                "input_feature_dim": input_dim,
                "output_dim": output_dim,
                "ceiling": ceiling,
                "batch_size": self.batch_size,
            }
            init_params.update(self.metric_params.get(metric_name, {}))
            metric = metric_cls(**init_params)

            # Compute
            res = metric.compute(
                extractor=extractor,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                test_dataloader=test_loader
            )
            results[metric_name] = res

        # 6. Save Results (Append Logic)
        timestamp = datetime.datetime.utcnow().isoformat()
        results['timestamp'] = timestamp

        # Filename based on the BENCHMARK CLASS NAME
        benchmark_name = self.__class__.__name__
        results_file = os.path.join(self.results_dir, f"{benchmark_name}.pkl")

        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    prev = pickle.load(f)

                # Check format of previous file
                prev_metrics = prev.get("metrics", [])

                # Ensure it's a list so we can append
                if isinstance(prev_metrics, dict):
                    prev_metrics = [prev_metrics]

                prev_metrics.append(results)
                merged = {"metrics": prev_metrics, "ceiling": ceiling}
                print(f"Appending results to existing file: {results_file}")
            except Exception as e:
                print(
                    f"Could not load existing results file, overwriting: {e}")
                merged = {"metrics": [results], "ceiling": ceiling}
        else:
            print(f"Creating new results file: {results_file}")
            merged = {"metrics": [results], "ceiling": ceiling}

        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)

        return results
