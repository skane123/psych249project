from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Callable, Union, Tuple, Any

import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch._dynamo import OptimizedModule

import wandb

from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import itertools  # For grid search

# Ensure these are in utils
from .utils import LinearInternalModel, AttentionPoolingInternalModel, TransformerInternalModel, pearson_correlation_scorer
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class FeatureNormalizer(nn.Module):
    """Handles different types of feature normalization strategies."""

    def __init__(self, feature_dim: int, normalization_type: str = "layer_norm",
                 momentum: float = 0.1, device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.normalization_type = normalization_type
        self.device = device

        if normalization_type == "layer_norm":
            self.norm = nn.LayerNorm(feature_dim, eps=1e-8)
        elif normalization_type == "batch_norm":
            self.norm = nn.BatchNorm1d(
                feature_dim, eps=1e-8, momentum=momentum)
        elif normalization_type == "running_stats":
            self.momentum = momentum
            self.register_buffer('running_mean', torch.zeros(feature_dim))
            self.register_buffer('running_var', torch.ones(feature_dim))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        elif normalization_type == "robust_zscore":
            # For robust z-score normalization using percentiles
            self.register_buffer('running_median', torch.zeros(feature_dim))
            self.register_buffer('running_mad', torch.ones(
                feature_dim))  # median absolute deviation
            self.momentum = momentum
        elif normalization_type in ["cosine", "unit_norm"]:
            pass  # No parameters needed
        else:
            raise ValueError(
                f"Unknown normalization_type: {normalization_type}")

        self.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features

        # Handle NaN and inf values
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.normalization_type == "layer_norm":
            return self.norm(features)

        elif self.normalization_type == "batch_norm":
            # Reshape for BatchNorm1d if needed
            original_shape = features.shape
            if features.dim() > 2:
                features = features.view(-1, features.size(-1))
            features = self.norm(features)
            return features.view(original_shape)

        elif self.normalization_type == "running_stats":
            if self.training:
                # Update running statistics
                batch_mean = features.mean(dim=0, keepdim=True)
                batch_var = features.var(dim=0, keepdim=True, unbiased=False)

                # Update running stats
                if self.num_batches_tracked == 0:
                    self.running_mean = batch_mean.squeeze(0)
                    self.running_var = batch_var.squeeze(0)
                else:
                    self.running_mean = (
                        1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze(0)
                    self.running_var = (
                        1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze(0)

                self.num_batches_tracked += 1

                # Normalize using batch stats during training
                return (features - batch_mean) / (torch.sqrt(batch_var) + 1e-8)
            else:
                # Use running stats during inference
                return (features - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)

        elif self.normalization_type == "robust_zscore":
            if self.training:
                # Update running median and MAD
                batch_median = torch.median(
                    features, dim=0, keepdim=True).values
                batch_mad = torch.median(
                    torch.abs(features - batch_median), dim=0, keepdim=True).values

                # Update running stats
                self.running_median = (
                    1 - self.momentum) * self.running_median + self.momentum * batch_median.squeeze(0)
                self.running_mad = (
                    1 - self.momentum) * self.running_mad + self.momentum * batch_mad.squeeze(0)

                # Normalize using batch stats
                # 1.4826 for normal distribution consistency
                return (features - batch_median) / (batch_mad * 1.4826 + 1e-8)
            else:
                return (features - self.running_median) / (self.running_mad * 1.4826 + 1e-8)

        elif self.normalization_type == "cosine":
            # L2 normalize features to unit sphere
            return torch.nn.functional.normalize(features, p=2, dim=-1, eps=1e-8)

        elif self.normalization_type == "unit_norm":
            # Scale to unit norm but keep direction
            norms = torch.norm(features, dim=-1, keepdim=True)
            return features / (norms + 1e-8)

        elif self.normalization_type == "batch_zscore":
            # Simple batch-wise z-score
            batch_mean = features.mean(dim=0, keepdim=True)
            batch_std = features.std(dim=0, keepdim=True)
            return (features - batch_mean) / (batch_std + 1e-8)

        else:
            return features


class OnlineMetric(ABC):
    def __init__(
        self,
        num_classes: int,  # For classifiers, 0 or 1 for regressors
        input_feature_dim: int,
        internal_model_type: str,  # "linear" or "transformer"
        internal_model_params: Optional[Dict[str, Any]] = None,
        lr_options: List[float] = [1e-3, 1e-4, 1e-5],
        wd_options: List[float] = [0],
        n_epochs: int = 1,
        patience: int = 10,  # For early stopping
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        ceiling: Optional[np.ndarray] = None,
        task_type: str = "classification",  # "classification" or "regression",
        wandb_project: Optional[str] = None,
        # NEW PARAMETERS FOR STABILIZATION
        # "layer_norm", "batch_norm", "running_stats", "robust_zscore", "cosine", "batch_zscore", "none"
        feature_normalization: str = "layer_norm",
        gradient_clip_norm: float = 1.0,
        # e.g., 5.0 to clip features to [-5, 5]
        feature_clip_value: Optional[float] = None,
        use_mixed_precision: bool = True,
        stabilize_loss: bool = True,
        # "wsd" (Warmup-Stable-Decay) or "cosine" (Warmup-Cosine)
        scheduler_type: str = "wsd",
    ):
        self.num_classes = num_classes
        self.input_feature_dim = input_feature_dim
        self.internal_model_type = internal_model_type
        self.internal_model_params = internal_model_params if internal_model_params else {}
        self.lr_options = lr_options
        self.wd_options = wd_options
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size  # Batch size for training the internal model
        self.device = device
        self.task_type = task_type
        self.wandb_project = wandb_project or os.environ.get("WANDB_PROJECT")

        # Stabilization parameters
        self.feature_normalization = feature_normalization
        self.gradient_clip_norm = gradient_clip_norm
        self.feature_clip_value = feature_clip_value
        self.use_mixed_precision = use_mixed_precision
        self.stabilize_loss = stabilize_loss
        self.scheduler_type = scheduler_type

        self.best_internal_model_state: Optional[Dict] = None
        self.best_hyperparams: Optional[Dict] = None
        self.best_val_score: float = - float('inf')

        self.ceiling = ceiling
        if self.ceiling is not None:
            self.ceiling = np.asarray(self.ceiling)
            self.ceiling[self.ceiling <= 1e-6] = 1e-6

        self._determine_internal_output_dim()

        # Initialize feature normalizer
        if self.feature_normalization != "none":
            self.feature_normalizer = FeatureNormalizer(
                self.input_feature_dim,
                self.feature_normalization,
                device=self.device
            )
        else:
            self.feature_normalizer = None

    def _determine_internal_output_dim(self):
        if self.task_type == "classification":
            if self.num_classes == 2:  # Binary classification
                self.internal_output_dim = 1  # For BCEWithLogitsLoss
            elif self.num_classes > 2:  # Multiclass classification
                self.internal_output_dim = self.num_classes
            else:  # Should not happen if num_classes is well defined
                raise ValueError(
                    "num_classes must be >= 2 for classification.")
        elif self.task_type == "regression":
            self.internal_output_dim = 1  # Assuming single target regression for now
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _get_internal_model(self) -> nn.Module:
        if self.internal_model_type == "linear":
            model = LinearInternalModel(
                self.input_feature_dim, self.internal_output_dim)
        elif self.internal_model_type == "attention":
            model = AttentionPoolingInternalModel(
                self.input_feature_dim, self.internal_output_dim)
        elif self.internal_model_type == "transformer":
            # TransformerInternalModel expects input_dim to be feature_dim from extractor
            # output_dim is num_classes or regression_dim
            merged_params = {"input_dim": self.input_feature_dim,
                             "output_dim": self.internal_output_dim}
            merged_params.update(self.internal_model_params)
            model = TransformerInternalModel(**merged_params)
        else:
            raise ValueError(
                f"Unknown internal_model_type: {self.internal_model_type}")
        return model.to(self.device)

    def _get_optimizer(self, model_params, lr: float, weight_decay: float) -> optim.Optimizer:
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)

    def _get_criterion(self) -> nn.Module:
        if self.task_type == "classification":
            if self.internal_output_dim == 1:  # Binary
                return nn.BCEWithLogitsLoss()
            else:  # Multiclass
                return nn.CrossEntropyLoss()
        elif self.task_type == "regression":
            return nn.MSELoss()
        else:  # Should not be reached
            raise ValueError(
                f"Cannot determine criterion for task_type: {self.task_type}")

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature normalization with optional clipping."""
        if features.numel() == 0:
            return features

        if features.dtype == torch.float64:
            features = features.float()

        # Clip features if specified
        if self.feature_clip_value is not None:
            features = torch.clamp(
                features, -self.feature_clip_value, self.feature_clip_value)

        # Apply normalization
        if self.feature_normalizer is not None:
            features = self.feature_normalizer(features)

        return features

    def _stabilize_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply loss stabilization techniques."""
        if not self.stabilize_loss:
            return loss

        # Handle NaN and extreme values in loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

        # Optional: clip loss to reasonable range
        loss = torch.clamp(loss, -10.0, 10.0)

        return loss

    def _unpack_labels(self, batch_labels: Any) -> torch.Tensor:
        """
        Unpacks labels from the dataloader batch to a tensor suitable for the loss function.
        Handles tuples like (stim_name, ground_truth_label, optional_behavioral_label)
        or simple label tensors.
        """
        if isinstance(batch_labels, tuple) and len(batch_labels) > 1:
            # Assuming the primary label is the second element if it's a tuple
            # e.g. (stimulus_ids_batch, labels_batch, optional_other_labels_batch)
            # or (stimulus_ids_batch, labels_batch)
            primary_labels = batch_labels[1]
        else:
            primary_labels = batch_labels

        if not isinstance(primary_labels, torch.Tensor):
            try:
                # Attempt conversion if it's list/numpy of numbers
                primary_labels = torch.tensor(primary_labels)
            except Exception as e:
                raise TypeError(
                    f"Could not convert batch_labels to tensor. Received: {type(batch_labels)}, Content: {batch_labels}, Error: {e}")

        labels_tensor = primary_labels.to(self.device)

        if self.task_type == "classification":
            if self.internal_output_dim == 1:  # Binary with BCEWithLogitsLoss
                return labels_tensor.float().unsqueeze(1)  # Ensure shape (batch_size, 1)
            else:  # Multiclass with CrossEntropyLoss
                return labels_tensor.long()  # Expects class indices
        elif self.task_type == "regression":
            return labels_tensor.float().unsqueeze(1) if labels_tensor.ndim == 1 else labels_tensor.float()
        return labels_tensor

    def train_and_evaluate(
        self,
        extractor: 'OnlineFeatureExtractor',  # Forward declaration
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Manages the grid search, training, and evaluation of the internal model.
        """
        # Create validation set if not provided
        current_train_loader = train_dataloader
        val_loader_internal = val_dataloader

        param_grid = list(itertools.product(self.lr_options, self.wd_options))

        grid_search_bar = tqdm(param_grid, desc="Grid Search Hyperparameters")
        for lr, wd in grid_search_bar:
            grid_search_bar.set_postfix({"lr": lr, "wd": wd})

            internal_model = self._get_internal_model()
            internal_model = nn.DataParallel(internal_model)
            optimizer = self._get_optimizer(
                internal_model.parameters(), lr, wd)

            # ─── SCHEDULER SETUP ────────────────────────
            total_steps = self.n_epochs * len(current_train_loader)

            # WSD Logic Setup
            # Standard decay length (20% or 1 epoch min)
            decay_epochs_count = max(5, int(self.n_epochs * 0.05))
            decay_triggered = False

            # Use a variable for max_epochs because WSD might extend the run
            current_max_epochs = self.n_epochs

            if self.scheduler_type == "cosine":
                # 5% Warmup then Cosine
                warmup_steps = int(0.05 * total_steps)
                cosine_steps = total_steps - warmup_steps

                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[
                        LinearLR(optimizer, start_factor=min(
                            1e-6, 1e-2/(lr*1000)), end_factor=1.0, total_iters=warmup_steps),
                        CosineAnnealingLR(
                            optimizer, T_max=cosine_steps, eta_min=0.0)
                    ],
                    milestones=[warmup_steps]
                )
            elif self.scheduler_type == "wsd":
                # Warmup then Constant (Infinite Stable Phase until patience hit)
                # 1 epoch warmup standard for WSD in this codebase
                warmup_steps = len(current_train_loader)

                warmup_sched = LinearLR(
                    optimizer,
                    start_factor=min(1e-6, 1e-2 / (lr * 1000)),
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )

                # Constant scheduler for effectively infinite steps until manually stopped/switched
                stable_sched = ConstantLR(
                    optimizer, factor=1.0, total_iters=999999999)

                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_sched, stable_sched],
                    milestones=[warmup_steps],
                )
            else:
                raise ValueError(
                    f"Unknown scheduler_type: {self.scheduler_type}")
            # ────────────────────────────────────────────────────────────────────

            # Initialize scaler for mixed precision if enabled
            scaler = torch.amp.GradScaler(
                'cuda') if self.use_mixed_precision else None
            criterion = self._get_criterion()

            current_best_val_epoch_score = - \
                float('inf') if self.task_type == "classification" else float('inf')
            current_best_model_state_epoch = None
            epochs_no_improve = 0

            # Set feature normalizer to training mode
            if self.feature_normalizer is not None:
                self.feature_normalizer.train()

            # Using while loop to allow WSD to extend training dynamically
            epoch = 0
            epoch_bar = tqdm(total=self.n_epochs,
                             desc=f"Training (lr={lr}, wd={wd})", leave=False)

            while epoch < current_max_epochs:
                internal_model.train()
                total_train_loss = 0.0

                train_preds, train_true = [], []

                batch_bar = tqdm(
                    current_train_loader, desc=f"Epoch {epoch+1}/{current_max_epochs}", leave=False)

                for batch_idx, (batch_data, batch_labels_raw) in enumerate(batch_bar):
                    # Handles device placement
                    optimizer.zero_grad()
                    labels = self._unpack_labels(batch_labels_raw)

                    # Use mixed precision if enabled
                    if self.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            if isinstance(batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    batch_data.to(self.device))
                            # Apply feature normalization
                            features = self._normalize_features(features)

                            outputs = internal_model(features)
                            outputs = torch.clamp(outputs, -10.0, +10.0)
                            loss = criterion(outputs, labels)
                            loss = self._stabilize_loss(loss)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            internal_model.parameters(), max_norm=self.gradient_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if isinstance(batch_data, dict):
                            features = extractor.extract_features_for_batch(
                                batch_data)
                        else:
                            features = extractor.extract_features_for_batch(
                                batch_data.to(self.device))

                        # Apply feature normalization
                        features = self._normalize_features(features)

                        outputs = internal_model(features)
                        outputs = torch.clamp(outputs, -10.0, +10.0)
                        loss = criterion(outputs, labels)
                        loss = self._stabilize_loss(loss)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            internal_model.parameters(), max_norm=self.gradient_clip_norm)
                        optimizer.step()

                    scheduler.step()
                    total_train_loss += loss.item()

                    if self.task_type == "classification":
                        if self.internal_output_dim == 1:
                            batch_preds = (torch.sigmoid(
                                outputs) > 0.5).float()
                        else:
                            batch_preds = torch.argmax(outputs, dim=1)
                        train_preds.append(batch_preds.detach().cpu())
                        train_true.append(labels.detach().cpu())
                        batch_metric = (
                            batch_preds == labels).float().mean().item()
                        batch_bar.set_postfix({
                            "batch_acc": f"{batch_metric:.4f}",
                            "loss": f"{loss.item():.4f}"
                        })
                    else:
                        # For regression, compute batch-level correlation and R²
                        train_preds.append(outputs.detach().cpu())
                        train_true.append(labels.detach().cpu())

                        # Compute batch Pearson correlation (Spearman-Brown corrected) and R²
                        if outputs.numel() > 1:
                            outputs_np = outputs.detach().cpu().numpy().ravel()
                            labels_np = labels.detach().cpu().numpy().ravel()

                            # Pearson correlation with Spearman-Brown correction
                            if outputs_np.std() > 1e-8 and labels_np.std() > 1e-8:
                                raw_corr = np.corrcoef(
                                    outputs_np, labels_np)[0, 1]
                                batch_corr = self._spearman_brown_correction(
                                    raw_corr)
                            else:
                                batch_corr = 0.0

                            # R² score
                            ss_res = np.sum((labels_np - outputs_np) ** 2)
                            ss_tot = np.sum(
                                (labels_np - labels_np.mean()) ** 2)
                            batch_r2 = 1 - \
                                (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
                        else:
                            batch_corr = 0.0
                            batch_r2 = 0.0

                        batch_bar.set_postfix({
                            "batch_corr": f"{batch_corr:.4f}",
                            "batch_r2": f"{batch_r2:.4f}",
                            "loss": f"{loss.item():.4f}"
                        })

                    # ─── W&B: log per batch ───────────────────────────
                    if wandb.run:
                        log_dict = {
                            "train/loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "grid_lr": lr,
                            "grid_wd": wd,
                        }

                        if self.task_type == "classification":
                            log_dict["train/accuracy"] = batch_metric
                        else:
                            log_dict["train/batch_corr"] = batch_corr
                            log_dict["train/batch_r2"] = batch_r2

                        wandb.log(log_dict, step=getattr(
                            self, "global_step", 0))
                        self.global_step = getattr(self, "global_step", 0) + 1
                    # ────────────────────────────────────────────────────

                avg_train_loss = total_train_loss / len(current_train_loader)

                train_preds_all = torch.cat(train_preds)
                train_true_all = torch.cat(train_true)

                if self.task_type == "classification":
                    train_accuracy = accuracy_score(
                        train_true_all.numpy().ravel(),
                        train_preds_all.numpy().ravel()
                    )
                else:
                    # Compute both correlation and R² for regression
                    from scipy.stats import pearsonr
                    y_true_np = train_true_all.numpy().ravel()
                    y_pred_np = train_preds_all.numpy().ravel()

                    if y_true_np.std() > 1e-8 and y_pred_np.std() > 1e-8:
                        # Correlation (Spearman-Brown corrected)
                        raw_corr, _ = pearsonr(y_true_np, y_pred_np)
                        train_accuracy = self._spearman_brown_correction(
                            raw_corr) if not np.isnan(raw_corr) else 0.0

                        # R²
                        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
                        ss_tot = np.sum((y_true_np - y_true_np.mean()) ** 2)
                        train_r2 = 1 - \
                            (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
                    else:
                        train_accuracy = 0.0
                        train_r2 = 0.0

                # Validation
                internal_model.eval()
                if self.feature_normalizer is not None:
                    self.feature_normalizer.eval()

                val_preds, val_true = [], []
                with torch.no_grad():
                    val_bar = tqdm(
                        val_loader_internal, desc=f"Epoch {epoch+1} Validation Cycle", leave=False)
                    for val_batch_data, val_batch_labels_raw in val_bar:
                        labels = self._unpack_labels(val_batch_labels_raw)

                        if self.use_mixed_precision:
                            with torch.no_grad(), torch.amp.autocast('cuda'):
                                if isinstance(val_batch_data, dict):
                                    features = extractor.extract_features_for_batch(
                                        val_batch_data)
                                else:
                                    features = extractor.extract_features_for_batch(
                                        val_batch_data.to(self.device))

                                # Apply feature normalization
                                features = self._normalize_features(features)
                                logits = internal_model(features)
                        else:
                            if isinstance(val_batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    val_batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    val_batch_data.to(self.device))

                            # Apply feature normalization
                            features = self._normalize_features(features)
                            logits = internal_model(features)

                        val_true.append(labels.cpu())
                        val_preds.append(logits.cpu())

                val_preds_all = torch.cat(val_preds)
                val_true_all = torch.cat(val_true)

                val_score = self._calculate_validation_score(
                    val_true_all, val_preds_all)

                # Unpack if regression (returns tuple)
                if self.task_type == "regression":
                    val_corr, val_r2 = val_score
                    val_score_for_comparison = val_corr  # Use correlation for model selection
                else:
                    val_corr = val_score
                    val_r2 = None
                    val_score_for_comparison = val_score

                if self.task_type == "classification":
                    epoch_bar.set_postfix({
                        "train_loss": f"{avg_train_loss:.4f}",
                        "train_acc": f"{train_accuracy:.4f}",
                        "val_acc": f"{val_score:.4f}"
                    })
                else:
                    epoch_bar.set_postfix({
                        "train_loss": f"{avg_train_loss:.4f}",
                        "train_corr": f"{train_accuracy:.4f}",
                        "train_r2": f"{train_r2:.4f}",
                        "val_corr": f"{val_corr:.4f}",
                        "val_r2": f"{val_r2:.4f}"
                    })

                # Check improvement
                improved = False
                if self.task_type == "classification":
                    if val_score_for_comparison > current_best_val_epoch_score:
                        improved = True
                else:  # Regression
                    # Using correlation (higher is better)
                    if val_score_for_comparison > current_best_val_epoch_score:
                        improved = True

                if improved:
                    current_best_val_epoch_score = val_score
                    if isinstance(internal_model, nn.DataParallel):
                        state = internal_model.module.state_dict()
                    else:
                        state = internal_model.state_dict()
                    current_best_model_state_epoch = state
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # ─── W&B: log per epoch ───────────────────────────────
                if wandb.run:
                    log_dict = {
                        "epoch/train_loss": avg_train_loss,
                        "epoch/lr": scheduler.get_last_lr()[0],
                    }

                    if self.task_type == "classification":
                        log_dict["epoch/train_accuracy"] = train_accuracy
                        log_dict["epoch/val_score"] = val_score
                    else:
                        log_dict["epoch/train_corr"] = train_accuracy
                        log_dict["epoch/train_r2"] = train_r2
                        log_dict["epoch/val_corr"] = val_corr
                        log_dict["epoch/val_r2"] = val_r2

                    wandb.log(log_dict, step=getattr(self, "global_step", 0))
                # ────────────────────────────────────────────────────

                # ─── EARLY STOPPING / DECAY TRIGGER LOGIC ─────────────
                if self.scheduler_type == "wsd":
                    # WSD Logic: If patience reached in stable phase, trigger decay phase
                    if not decay_triggered and epochs_no_improve >= self.patience:
                        print(
                            f"\n[WSD] Patience reached at epoch {epoch+1}. Switching to decay phase for {decay_epochs_count} epochs.")

                        # 1. Load best model found so far
                        if current_best_model_state_epoch is not None:
                            if isinstance(internal_model, nn.DataParallel):
                                internal_model.module.load_state_dict(
                                    current_best_model_state_epoch)
                            else:
                                internal_model.load_state_dict(
                                    current_best_model_state_epoch)

                        # 2. Switch Scheduler to Linear Decay (1.0 -> 0.0)
                        # The optimizer still holds the base LR. We just need a scheduler that decays it.
                        decay_steps = decay_epochs_count * \
                            len(current_train_loader)
                        scheduler = LinearLR(
                            optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)

                        # 3. Extend the loop and reset state for decay
                        decay_triggered = True
                        current_max_epochs = epoch + 1 + decay_epochs_count
                        epoch_bar.total = current_max_epochs
                        epochs_no_improve = 0  # Disable patience check during decay

                    elif not decay_triggered and epoch >= self.n_epochs - 1:
                        # Case: Ran out of normal epochs without triggering patience.
                        # Do we trigger decay or stop?
                        # To ensure best performance, WSD usually decays at the end if not triggered earlier.
                        # But strict interpretation of "decay on patience" might imply we stop.
                        # Let's stop to avoid infinite loops if patience is huge.
                        break

                    elif decay_triggered and epoch >= current_max_epochs - 1:
                        # Decay phase finished
                        break

                else:
                    # Standard (Cosine / Other) Logic: Stop on patience
                    if epochs_no_improve >= self.patience:
                        break

                epoch += 1
                epoch_bar.update(1)

            epoch_bar.close()

            # Update best overall model from grid search
            if self.task_type == "classification":
                if current_best_val_epoch_score > self.best_val_score:
                    self.best_val_score = current_best_val_epoch_score
                    self.best_internal_model_state = current_best_model_state_epoch
                    self.best_hyperparams = {"lr": lr, "wd": wd}
            else:  # Regression
                if current_best_val_epoch_score < self.best_val_score:
                    self.best_val_score = current_best_val_epoch_score
                    self.best_internal_model_state = current_best_model_state_epoch
                    self.best_hyperparams = {"lr": lr, "wd": wd}

        grid_search_bar.close()

        # Evaluate the best model on the (optional) test set
        final_scores = {}
        if test_dataloader and self.best_internal_model_state:
            best_model = self._get_internal_model()
            best_model.load_state_dict(self.best_internal_model_state)
            best_model = nn.DataParallel(best_model)

            best_model.eval()
            if self.feature_normalizer is not None:
                self.feature_normalizer.eval()

            test_preds_all, test_true_all, test_stim_ids_all = [], [], []
            with torch.no_grad():
                test_batch_bar = tqdm(
                    test_dataloader, desc="Evaluating on Test Set", leave=False)
                for test_batch_data, test_batch_labels_raw in test_batch_bar:
                    stim_ids_batch = None
                    if hasattr(test_batch_labels_raw, "__len__") and len(test_batch_labels_raw) == 3:
                        stim_ids_batch, test_batch_labels, test_behavioral_target = test_batch_labels_raw
                    else:
                        test_batch_labels = test_batch_labels_raw

                    if self.use_mixed_precision:
                        with torch.no_grad(), torch.amp.autocast('cuda'):
                            if isinstance(test_batch_data, dict):
                                features = extractor.extract_features_for_batch(
                                    test_batch_data)
                            else:
                                features = extractor.extract_features_for_batch(
                                    test_batch_data.to(self.device))

                            # Apply feature normalization
                            features = self._normalize_features(features)
                            outputs = best_model(features)
                    else:
                        if isinstance(test_batch_data, dict):
                            features = extractor.extract_features_for_batch(
                                test_batch_data)
                        else:
                            features = extractor.extract_features_for_batch(
                                test_batch_data.to(self.device))

                        # Apply feature normalization
                        features = self._normalize_features(features)
                        outputs = best_model(features)

                    # This gets the primary GT label
                    labels = self._unpack_labels(test_batch_labels)

                    test_preds_all.append(outputs.cpu())
                    test_true_all.append(labels.cpu())  # Primary GT
                    if stim_ids_batch:
                        test_stim_ids_all.extend(stim_ids_batch)

                    # Check for behavioral target in test_batch_labels_raw
                    if hasattr(test_batch_labels_raw, "__len__") and len(test_batch_labels_raw) == 3:
                        if 'behavioral_gt' not in final_scores:
                            final_scores['behavioral_gt'] = []
                        final_scores['behavioral_gt'].append(
                            test_behavioral_target)

            test_preds_all = torch.cat(test_preds_all)
            test_true_all = torch.cat(test_true_all)
            if final_scores.get('behavioral_gt'):
                final_scores['behavioral_gt'] = torch.cat(final_scores['behavioral_gt']) if isinstance(
                    final_scores['behavioral_gt'][0], torch.Tensor) else torch.tensor(np.concatenate(final_scores['behavioral_gt']))

            final_scores.update(self._calculate_final_scores(
                test_true_all, test_preds_all, prefix="gt_"))

            if self.internal_output_dim == 1:  # Binary
                final_scores['preds'] = (torch.sigmoid(
                    test_preds_all) > 0.5).float().numpy().ravel()
            else:
                if self.task_type == "classification":
                    final_scores['preds'] = torch.argmax(
                        test_preds_all, dim=1).numpy().ravel()
                else:
                    final_scores['preds'] = test_preds_all

            final_scores['gt'] = test_true_all.numpy()
            if test_stim_ids_all:
                final_scores['stimulus'] = test_stim_ids_all

            if 'behavioral_gt' in final_scores and final_scores['behavioral_gt'] is not None:
                # Assuming behavioral_gt is already a tensor or can be converted
                human_scores = self._calculate_final_scores(
                    final_scores['behavioral_gt'], test_preds_all, prefix="human_")
                final_scores.update(human_scores)

        final_scores['best_hyperparams'] = self.best_hyperparams
        final_scores['best_val_score_during_grid_search'] = self.best_val_score
        return final_scores

    def _spearman_brown_correction(self, r: float, n: float = 2.0) -> float:
        """Apply Spearman-Brown correction to correlation coefficient."""
        if r <= 0 or n <= 0:
            return r
        r = float(np.clip(r, -0.999999, 0.999999))
        return (n * r) / (1.0 + (n - 1.0) * r)

    def _calculate_validation_score(self, y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> Union[float, Tuple[float, float]]:
        """
        Calculates validation score for model selection.

        For classification: accuracy
        For regression: (mean Pearson correlation, mean R²) - returns tuple

        Returns:
            float or tuple: Validation score (higher is better)
        """
        if self.task_type == "classification":
            if self.internal_output_dim == 1:  # Binary
                preds = (torch.sigmoid(y_pred_logits) > 0.5).float()
                return accuracy_score(y_true.numpy(), preds.numpy())
            else:  # Multiclass
                preds = torch.argmax(y_pred_logits, dim=1)
                return accuracy_score(y_true.numpy(), preds.numpy())

        elif self.task_type == "regression":
            from scipy.stats import pearsonr
            from sklearn.metrics import r2_score

            y_true_np = y_true.numpy()
            y_pred_np = y_pred_logits.numpy()

            def _spearman_brown(r: float, n: float = 2.0) -> float:
                if r <= 0 or n <= 0:
                    return r
                r = float(np.clip(r, -0.999999, 0.999999))
                return (n * r) / (1.0 + (n - 1.0) * r)

            if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
                # Multi-target: compute correlation and R² for each output dimension
                correlations = []
                r2_scores = []

                for i in range(y_true_np.shape[1]):
                    if y_true_np[:, i].std() < 1e-8 or y_pred_np[:, i].std() < 1e-8:
                        correlations.append(0.0)
                        r2_scores.append(0.0)
                    else:
                        r, _ = pearsonr(y_true_np[:, i], y_pred_np[:, i])
                        correlations.append(_spearman_brown(
                            r) if not np.isnan(r) else 0.0)

                        ss_res = np.sum(
                            (y_true_np[:, i] - y_pred_np[:, i]) ** 2)
                        ss_tot = np.sum(
                            (y_true_np[:, i] - y_true_np[:, i].mean()) ** 2)
                        r2_scores.append(1 - (ss_res / ss_tot)
                                         if ss_tot > 1e-8 else 0.0)

                mean_corr = np.mean(correlations)
                median_corr = np.median(correlations)
                mean_r2 = np.mean(r2_scores)
                median_r2 = np.median(r2_scores)

                # Log both metrics to W&B if available
                if wandb.run:
                    wandb.log({
                        "val/mean_pearson": mean_corr,
                        "val/median_pearson": median_corr,
                        "val/mean_r2": mean_r2,
                        "val/median_r2": median_r2,
                    }, step=getattr(self, "global_step", 0))

                # Return tuple: (correlation, r2) for model selection
                return (median_corr, median_r2)
            else:
                # Single target regression
                y_true_flat = y_true_np.ravel()
                y_pred_flat = y_pred_np.ravel()

                if y_true_flat.std() < 1e-8 or y_pred_flat.std() < 1e-8:
                    return (0.0, 0.0)

                r, _ = pearsonr(y_true_flat, y_pred_flat)
                corr = _spearman_brown(r) if not np.isnan(r) else 0.0

                ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
                ss_tot = np.sum((y_true_flat - y_true_flat.mean()) ** 2)
                val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

                if wandb.run:
                    wandb.log({
                        "val/pearson": corr,
                        "val/r2": val_r2,
                    }, step=getattr(self, "global_step", 0))

                return (corr, val_r2)

        return 0.0

    def _calculate_final_scores(self, y_true: torch.Tensor, y_pred_logits: torch.Tensor, prefix="") -> Dict[str, float]:
        """Calculates multiple scores for final reporting."""
        scores = {}
        y_true_np = y_true.numpy()

        if self.task_type == "classification":
            if self.internal_output_dim == 1:  # Binary
                y_pred_probs = torch.sigmoid(y_pred_logits).numpy()
                y_pred_labels = (y_pred_probs > 0.5).astype(int)
                y_true_np = (y_true_np > 0.5).astype(int)
                scores[f'{prefix}accuracy'] = accuracy_score(
                    y_true_np, y_pred_labels)
                scores[f'{prefix}f1_score'] = f1_score(
                    y_true_np, y_pred_labels, average='binary' if self.num_classes == 2 else 'macro', zero_division=0)
            else:  # Multiclass
                y_pred_labels = torch.argmax(y_pred_logits, dim=1).numpy()
                scores[f'{prefix}accuracy'] = accuracy_score(
                    y_true_np, y_pred_labels)
                scores[f'{prefix}f1_score'] = f1_score(
                    y_true_np, y_pred_labels, average='macro', zero_division=0)  # Use macro for multiclass
        elif self.task_type == "regression":
            y_pred_np = y_pred_logits.numpy()
            scores[f'{prefix}mse'] = mean_squared_error(y_true_np, y_pred_np)
            scores[f'{prefix}r2_score_uniform'] = r2_score(
                y_true_np, y_pred_np, multioutput='uniform_average')
            scores[f'{prefix}r2_score_weighted'] = r2_score(
                y_true_np, y_pred_np, multioutput='variance_weighted')
            # Calculate Pearson correlation for each output dimension if y_true_np is 2D
            if y_true_np.ndim > 1 and y_true_np.shape[1] > 1:
                pearson_coeffs = [pearsonr(y_true_np[:, i], y_pred_np[:, i])[
                    0] for i in range(y_true_np.shape[1])]
                pearson_coeffs = [_spearman_brown(r) if not np.isnan(
                    r) else 0.0 for r in pearson_coeffs]
                scores[f'{prefix}pearson_corr_mean'] = np.mean(pearson_coeffs)
                scores[f'{prefix}pearson_corr_median'] = np.median(
                    pearson_coeffs)
                scores[f'{prefix}pearson_corr_raw'] = np.array(pearson_coeffs)
                r2_scores = r2_score(y_true_np, y_pred_np,
                                     multioutput='raw_values')
                scores[f'{prefix}r2_mean_score'] = np.mean(r2_scores)
                scores[f'{prefix}r2_median_score'] = np.median(r2_scores)
            else:
                scores[f'{prefix}pearson_corr'] = self._spearman_brown_correction(pearsonr(
                    y_true_np.ravel(), y_pred_np.ravel())[0])
                scores[f'{prefix}r2_score'] = r2_score(
                    y_true_np, y_pred_np, multioutput='raw_values')

        return scores

    def apply_ceiling(self, scores_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.ceiling is None:
            return scores_dict

        ceiled_scores = {}
        for key, value in scores_dict.items():
            # Apply ceiling only to scalar metrics that are not hyperparams or raw data lists
            if isinstance(value, (float, np.floating, np.integer)) and \
               key not in ['best_val_score_during_grid_search'] and \
               not key.startswith('raw_') and \
               'hyperparams' not in key and \
               'preds' not in key and 'gt' not in key and 'stimulus' not in key and \
               'behavioral_gt' not in key:

                # If ceiling is an array, we need a strategy. For now, let's assume
                # if it's a single score metric, we use the mean of the ceiling.
                # This part might need refinement based on how ceiling is structured for online metrics.
                current_ceiling = np.mean(self.ceiling) if isinstance(
                    self.ceiling, np.ndarray) else self.ceiling
                if current_ceiling != 0:
                    ceiled_scores[f"ceiled_{key}"] = value / current_ceiling
                else:
                    # Avoid division by zero
                    ceiled_scores[f"ceiled_{key}"] = value
            elif isinstance(value, np.ndarray) and value.ndim == 1 and \
                    key not in ['preds', 'gt', 'stimulus', 'behavioral_gt'] and \
                    'pearson_corr_all' in key:  # Special case for per-target Pearson
                if self.ceiling.shape == value.shape:
                    ceiled_value = np.zeros_like(value)
                    valid_ceiling_mask = self.ceiling != 0
                    ceiled_value[valid_ceiling_mask] = value[valid_ceiling_mask] / \
                        self.ceiling[valid_ceiling_mask]
                    # keep original where ceiling is 0
                    ceiled_value[~valid_ceiling_mask] = value[~valid_ceiling_mask]
                    ceiled_scores[f"ceiled_{key}"] = ceiled_value
                else:
                    print(
                        f"Warning: Ceiling shape {self.ceiling.shape} does not match value shape {value.shape} for key {key}. Skipping ceiled score.")
                    # Keep original if shapes don't match
                    ceiled_scores[key] = value
            else:
                ceiled_scores[key] = value
        return ceiled_scores

    @abstractmethod
    def compute_raw(
        self,
        extractor: 'OnlineFeatureExtractor',
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:  # Changed from Union[Dict, float] to Dict
        pass

    def compute(
        self,
        extractor: 'OnlineFeatureExtractor',
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        wandb_project = self.wandb_project or os.environ.get('WANDB_PROJECT')

        # Sweep mode
        if wandb_project:
            # 1) unwrap your model's real class name
            base_mod = (
                extractor.model._orig_mod
                if isinstance(extractor.model, OptimizedModule)
                else extractor.model
            )
            model_name = base_mod.__class__.__name__

            # 2) build your sweep name exactly as before
            lr_str = "_".join(str(lr) for lr in self.lr_options)
            wd_str = "_".join(str(wd) for wd in self.wd_options)
            prefix = f"{model_name}__{self.internal_model_type}__sweep"
            sweep_name = f"{prefix}__lrs_{lr_str}__wds_{wd_str}"

            # 3) create the sweep config, passing model_name (a string), not the module
            sweep_config = {
                "name": sweep_name,
                "method": "grid",
                "metric": {
                    "name": "epoch/val_score",
                    "goal": "maximize" if self.task_type == "classification" else "minimize"
                },
                "parameters": {
                    "lr":               {"values": self.lr_options},
                    "wd":               {"values": self.wd_options},
                    "n_epochs":         {"values": [self.n_epochs]},
                    "batch_size":       {"values": [train_dataloader.batch_size]},
                    "task_type":        {"values": [self.task_type]},
                    "num_classes":      {"values": [self.num_classes]},
                    "feature_extractor": {"values": [model_name]},
                    "internal_model":   {"values": [self.internal_model_type]},
                    "feature_normalization": {"values": [self.feature_normalization]},
                    "scheduler_type": {"values": [self.scheduler_type]},
                },
            }

            sweep_id = wandb.sweep(sweep_config, project=wandb_project)
            all_results = {}
            grid_size = len(self.lr_options) * len(self.wd_options)

            def _sweep_run():
                # grab this run's config
                cfg = wandb.config

                # 4) build a human-readable run name and init with it
                _ = wandb.init(
                    project=wandb_project,
                    reinit=True,
                    job_type="sweep"
                )

                cfg = wandb.config
                run_name = (
                    f"{cfg.feature_extractor}"
                    f"__{cfg.internal_model}"
                    f"__lr_{cfg.lr}"
                    f"__wd_{cfg.wd}"
                    f"__norm_{cfg.feature_normalization}"
                )

                # now re-init to set the name
                run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=cfg,
                    reinit=True,
                    job_type="sweep",
                )
                # force training on exactly this pair
                self.lr_options = [cfg.lr]
                self.wd_options = [cfg.wd]

                # do the work and log metrics
                res = self.compute_raw(
                    extractor,
                    train_dataloader,
                    val_dataloader,
                    test_dataloader
                )
                wandb.log(res)
                wandb.finish()

                # save for post-processing
                all_results[(cfg.lr, cfg.wd)] = res

            # 5) launch your sweep agent in-process
            wandb.agent(sweep_id, function=_sweep_run, count=grid_size)

            # pick best run
            raw_scores_dict = self._select_best(all_results, self.task_type)
        else:
            raw_scores_dict = self.compute_raw(
                extractor, train_dataloader, val_dataloader, test_dataloader)

        if wandb_project:
            wandb.finish()

        final_scores = self.apply_ceiling(raw_scores_dict)
        return final_scores

    def _select_best(
        self,
        results: Dict[Tuple[float, float], Dict[str, Any]],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Pick the best metrics dict from a mapping of (lr,wd) -> metrics, based on validation score.
        """
        if not results:
            return {}
        if task_type == "classification":
            # Maximize best_val_score_during_grid_search
            best = max(
                results.items(),
                key=lambda kv: kv[1].get(
                    'best_val_score_during_grid_search', -float('inf'))
            )
        else:
            # Minimize best_val_score_during_grid_search for regression
            best = min(
                results.items(),
                key=lambda kv: kv[1].get(
                    'best_val_score_during_grid_search', float('inf'))
            )
        return best[1]
