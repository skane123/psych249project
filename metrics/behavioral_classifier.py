import itertools
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, KFold
from torch.utils.data import DataLoader, TensorDataset
from typing import Sequence, Optional, Tuple, List, Dict, Any, Union, Callable
from sklearn.metrics import accuracy_score, f1_score

from .base import BaseMetric
from .utils import pearson_correlation_scorer

from tqdm import tqdm


class BehavioralRegressionMetric(BaseMetric):
    def __init__(
        self,
        alpha_options: List[float] = [
            1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100
        ],
        ceiling: Optional[float] = None,
        mode: Optional[str] = "sklearn",
    ):
        super().__init__(ceiling)
        self.alpha_options = alpha_options
        self.mode = mode

    def _compile_results(
        self,
        y_pred: np.ndarray,
        y_val: np.ndarray,
        scores: Dict[str, np.ndarray],
        scoring_funcs: Dict[str, Callable],
        tag: Optional[str] = ""
    ) -> Dict[str, np.ndarray]:

        for name in scoring_funcs.keys():
            scores[tag+name] = []

        # Compute scores for each metric
        for name, scoring_func in scoring_funcs.items():
            fold_score = scoring_func(y_val, y_pred)

            if isinstance(fold_score, (float, int, np.number)):
                scores[tag+name].append(fold_score)
            else:
                scores[tag+name].append(np.array(scoring_func(y_val, y_pred)))
        return scores

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: np.ndarray,
        test_target: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        # target data expects the follwing formats: (str, int, int), (str, int), (int, int), (int)
        # The str is always going to be the stimulus name
        # The int will be either a ground truth or a behavioral target
        train_stim, y_train, _ = self._unpack_targets(target)
        test_stim,  y_test,  behavioral_target = self._unpack_targets(
            test_target)

        scoring_funcs = {
            "accuracy": lambda y_true, y_pred: accuracy_score(np.rint(y_true).astype(float), y_pred),
            "cohen_kappa": lambda y_true, y_pred: cohen_kappa_score(np.rint(y_true).astype(float), y_pred),
        }

        if self.mode == "sklearn":
            def model_factory():
                return ProbabilitiesClassifier(source, y_train, c_values=self.alpha_options)
        else:
            def model_factory():
                return TorchRProbabilitiesClassifier(source, y_train, test_source, y_test, behavioral_target)

        model = model_factory()
        if model is not None:
            model.fit()
            y_pred = model.predict(test_source)
            y_probs = model.predict_proba(test_source)

        scores = {}
        scores['preds'] = y_pred
        scores['probs'] = y_probs
        scores['gt'] = y_test
        if test_stim:
            scores['stimulus'] = test_stim

        # Compute scores for each metric
        scores = self._compile_results(
            y_pred, y_test, scores, scoring_funcs, "gt_")
        if behavioral_target is not None:
            scores['behavioral_gt'] = behavioral_target
            scores = self._compile_results(y_pred, behavioral_target,
                                           scores, scoring_funcs, "human_")

        return {name: np.array(score_list) for name, score_list in scores.items()}

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        return self.compute_raw(source, target, test_source, test_target)

    def _unpack_targets(
        self,
        target_entries: Sequence[Union[
            Tuple[str, float, float],
            Tuple[str, float],
            Tuple[int, float],
            int
        ]]
    ) -> Tuple[Optional[List[str]], np.ndarray, Optional[np.ndarray]]:
        """
        Turn a list of target_entries—each of which may be:
          (str, int, int), (str, int), (int, int), or int
        into:
          stimuli:  List[str] or None
          y:        np.ndarray of ints
          beh:      np.ndarray of ints or None
        """
        stimuli: List[Optional[str]] = []
        y_vals:   List[float] = []
        beh_vals: List[Optional[float]] = []

        for entry in target_entries:
            # normalize everything into a tuple
            if not isinstance(entry, (list, tuple)):
                entry = (entry,)

            # unpack by length
            if len(entry) == 3:
                stim, gt, bt = entry
            elif len(entry) == 2:
                first, second = entry
                if isinstance(first, str):
                    stim, gt, bt = first, second, None
                else:
                    stim, gt, bt = None, first, second
            elif len(entry) == 1:
                stim, gt, bt = None, entry[0], None
            else:
                raise ValueError(f"Unsupported target format: {entry!r}")

            stimuli.append(stim)
            y_vals.append(float(gt))
            beh_vals.append(None if bt is None else float(bt))

        # convert to numpy arrays
        y_arr = np.array(y_vals, dtype=float)

        # stimuli: drop if completely unused
        if all(s is None for s in stimuli):
            stimuli_out = None
        else:
            # safe to cast because all non‐None are str
            stimuli_out = [s or "" for s in stimuli]

        # behavioral: only if *any* entry had a bt
        if any(b is not None for b in beh_vals):
            # at this point every slot either has int or None,
            # but we *expect* a full parallel array, so we can
            # replace None→0 or better yet raise if there's a mismatch.
            if any(b is None for b in beh_vals):
                raise ValueError(
                    "Mixed presence of behavioral targets in your batch."
                )
            beh_arr = np.array(beh_vals, dtype=float)
        else:
            beh_arr = None

        return stimuli_out, y_arr, beh_arr


class TorchBehavioralRegressionMetric(BehavioralRegressionMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling=ceiling, mode='torch')


class ProbabilitiesClassifier:
    def __init__(self, source, target,
                 c_values=None, k_folds=5, scoring='accuracy', solver='lbfgs'):
        """
        Initialize the classifier with k-fold cross-validation and grid search for C parameter.

        Parameters:
        -----------
        c_values : list or None
            List of C values to try in the grid search. If None, uses default values.
        k_folds : int
            Number of folds for cross-validation.
        scoring : str
            Scoring metric for grid search. Default is 'accuracy'.
        solver: str
            Solver for logistic regression model.
        """
        self.X = source
        self.Y = target
        # Default C values if none provided
        if c_values is None:
            c_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

        # Create the base classifier
        if solver:
            base_classifier = LogisticRegression(solver=solver)
        else:
            base_classifier = LogisticRegression(solver=solver)

        # Set up k-fold cross-validation
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Create the grid search with the classifier and parameters
        self._grid_search = GridSearchCV(
            estimator=base_classifier,
            param_grid={'C': c_values},
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=3
        )

        self._scaler = None
        self._best_c = None

    def fit(self):
        """
        Fit the model using grid search to find the best C value.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Fit and transform with the scaler
        self._scaler = StandardScaler().fit(self.X)
        X_scaled = self._scaler.transform(self.X)

        # Perform grid search to find the best C value
        self._grid_search.fit(X_scaled, self.Y)

        # Store the best C value for reference
        self._best_c = self._grid_search.best_params_['C']

        # Print the results
        print(f"Best C value found: {self._best_c}")
        print(
            f"Best cross-validation score: {self._grid_search.best_score_:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions using the best model from grid search.

        Parameters:
        -----------
        X : array-like
            Features matrix to predict.

        Returns:
        --------
        predictions : array
            Predicted labels.
        """
        assert len(X.shape) == 2, "expected 2-dimensional input"
        X_scaled = self._scaler.transform(X)
        preds = self._grid_search.predict(X_scaled)
        return preds

    def predict_proba(self, X):
        """
        Get probability estimates for each class.

        Parameters:
        -----------
        X : array-like
            Features matrix.

        Returns:
        --------
        probabilities : array
            Probability estimates.
        """
        assert len(X.shape) == 2, "expected 2-dimensional input"
        X_scaled = self._scaler.transform(X)
        probs = self._grid_search.predict_proba(X_scaled)
        return probs

    def get_best_c(self):
        """
        Get the best C value found by grid search.

        Returns:
        --------
        best_c : float
            The best C value.
        """
        if self._best_c is None:
            raise ValueError("Model has not been fitted yet.")
        return self._best_c

    def get_cv_results(self):
        """
        Get the cross-validation results.

        Returns:
        --------
        cv_results : dict
            Cross-validation results.
        """
        return self._grid_search.cv_results_


class SimpleNet(nn.Module):
    """Simple neural network with L2 regularization support."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.penalty = 0.0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def set_penalty(self, penalty: float):
        self.penalty = penalty

    def get_penalty_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.penalty * l2_loss


class TorchRProbabilitiesClassifier:
    """
    A PyTorch-based classifier for probability estimation with regression.
    Includes grid search for hyperparameters and supports multi-GPU training.
    Handles 1-D targets for both binary and multiclass classification.
    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: np.ndarray,
        test_target: np.ndarray,
        behavioral_target: Optional[np.ndarray] = None,
        hidden_size: int = 128,
        # lr_options: List[float] = [1e-4, 1e-3, 1e-2],
        # penalty_options: List[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1],
        # batch_size: int = 32,
        # n_epochs: int = 50,
        # patience: int = 50,
        lr_options: List[float] = [1e-3],
        penalty_options: List[float] = [1e-3, 1e-2],
        batch_size: int = 512,
        n_epochs: int = 10,
        patience: int = 5,
        use_multi_gpu: bool = False
    ):
        """
        Initialize the TorchRProbabilitiesClassifier with grid search capabilities.

        Args:
            source: Training features
            target: Training labels/targets (1-D array)
            test_source: Test features
            test_target: Test labels/targets (1-D array)
            behavioral_target: Optional additional behavioral targets
            hidden_size: Size of hidden layers in the neural network
            lr_options: List of learning rates to try in grid search
            penalty_options: List of L2 regularization strengths to try
            batch_size: Batch size for training
            n_epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
            use_multi_gpu: Whether to use multiple GPUs if available
        """
        self.source = torch.tensor(source, dtype=torch.float32)

        # Handle 1-D targets properly
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if test_target.ndim == 1:
            test_target = test_target.reshape(-1, 1)

        self.target = torch.tensor(target, dtype=torch.float32)
        self.test_source = torch.tensor(test_source, dtype=torch.float32)
        self.test_target = torch.tensor(test_target, dtype=torch.float32)

        if behavioral_target is not None:
            if behavioral_target.ndim == 1:
                behavioral_target = behavioral_target.reshape(-1, 1)
            self.behavioral_target = torch.tensor(
                behavioral_target, dtype=torch.float32)
        else:
            self.behavioral_target = None

        self.input_size = source.shape[1]
        self.hidden_size = hidden_size
        self.lr_options = lr_options
        self.penalty_options = penalty_options
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience

        # Determine the task type and output size
        self.is_binary, self.output_size, self.num_classes, self.class_mapping = self._analyze_target(
            target)

        # Setup device configuration
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if self.use_multi_gpu else 1

        if torch.cuda.is_available():
            print(
                f"Using {self.num_gpus} GPU(s): {', '.join([torch.cuda.get_device_name(i) for i in range(self.num_gpus)])}")
        else:
            print("Using CPU for training")

        # Best model parameters from grid search
        self.best_model = None
        self.best_lr = None
        self.best_penalty = None
        self.best_score = -float('inf')

        # Flag to track if model has been trained
        self.is_fitted = False

    def _analyze_target(self, y: np.ndarray) -> Tuple[bool, int, int, Optional[Dict]]:
        """
        Analyze target array to determine task type and output configuration.
        Handles missing class labels by creating a mapping from present classes to indices.

        Args:
            y: Target array (can be 1-D or 2-D)

        Returns:
            Tuple of (is_binary, output_size, num_classes, class_mapping)
        """
        # Handle 1-D targets
        if y.ndim == 1:
            unique_values = np.unique(y)

            # Check if it's binary classification (values between 0 and 1)
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                return True, 1, 2, None

            # Check if it's continuous binary (values between 0 and 1)
            elif np.all((y >= 0) & (y <= 1)) and len(unique_values) > 2:
                return True, 1, 2, None

            # Multiclass classification - handle missing labels
            else:
                unique_classes = np.sort(unique_values)
                max_class = int(np.max(unique_classes))
                num_present_classes = len(unique_classes)

                # Check if we have missing classes (gaps in class indices)
                expected_classes = np.arange(max_class + 1)
                missing_classes = set(expected_classes) - set(unique_classes)

                if len(missing_classes) > 0:
                    print(
                        f"Warning: Missing {len(missing_classes)} classes out of {max_class + 1} total classes")
                    print(f"Present classes: {unique_classes.tolist()}")
                    print(f"Missing classes: {sorted(list(missing_classes))}")

                    # Create mapping from original class indices to contiguous indices
                    class_mapping = {orig_class: new_idx for new_idx,
                                     orig_class in enumerate(unique_classes)}
                    # Also create reverse mapping for prediction output
                    reverse_mapping = {
                        new_idx: orig_class for orig_class, new_idx in class_mapping.items()}
                    class_mapping['reverse'] = reverse_mapping

                    return False, num_present_classes, num_present_classes, class_mapping
                else:
                    return False, max_class + 1, max_class + 1, None

        # Handle 2-D targets (legacy support)
        else:
            if y.shape[1] == 1:
                return self._analyze_target(y.ravel())
            elif y.shape[1] == 2 and np.all(np.sum(y, axis=1) == 1):
                return True, 2, 2, None
            else:
                return False, y.shape[1], y.shape[1], None

    def _get_loss_function(self) -> nn.Module:
        """
        Get appropriate loss function based on task type.

        Returns:
            PyTorch loss function
        """
        if self.is_binary:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def _build_model(self) -> nn.Module:
        """
        Build the neural network model.

        Returns:
            PyTorch model
        """
        model = SimpleNet(self.input_size, self.hidden_size, self.output_size)

        # Use DataParallel for multi-GPU training
        if self.use_multi_gpu:
            model = nn.DataParallel(model)

        return model.to(self.device)

    def _prepare_targets_for_loss(self, batch_y: torch.Tensor) -> torch.Tensor:
        """
        Prepare targets for loss calculation based on task type.
        Handles missing class labels by mapping to contiguous indices.

        Args:
            batch_y: Batch of targets

        Returns:
            Prepared targets for loss calculation
        """
        if self.is_binary:
            # For binary classification, keep as is (values between 0-1)
            return batch_y
        else:
            # For multiclass, convert to class indices if necessary
            if batch_y.shape[1] == 1:
                # Convert class labels using mapping if available
                if self.class_mapping is not None:
                    # Map original class indices to contiguous indices
                    mapped_targets = torch.zeros_like(
                        batch_y, dtype=torch.long)
                    for i, target in enumerate(batch_y):
                        original_class = int(target.item())
                        mapped_targets[i] = self.class_mapping[original_class]
                    return mapped_targets.squeeze()
                else:
                    # Already class indices
                    return batch_y.long().squeeze()
            else:
                # One-hot encoded, convert to indices
                return torch.argmax(batch_y, dim=1)

    def _train_evaluate_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        penalty: float
    ) -> Tuple[nn.Module, float]:
        """
        Train and evaluate a model with the given hyperparameters.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            criterion: Loss function
            train_loader: DataLoader for training data
            penalty: L2 regularization strength

        Returns:
            Tuple of (trained model, validation score)
        """
        # Set the penalty value in the model
        if self.use_multi_gpu:
            model.module.set_penalty(penalty)
        else:
            model.set_penalty(penalty)

        best_val_score = -float('inf')
        best_model_state = None
        no_improve_epochs = 0

        # Progress bar for epochs
        epoch_pbar = tqdm(range(self.n_epochs), desc="Training", leave=False)

        for epoch in epoch_pbar:
            # Training
            model.train()
            train_loss = 0.0

            # Progress bar for training batches
            train_pbar = tqdm(
                train_loader, desc=f"Epoch {epoch+1} - Train", leave=False)
            for batch_X, batch_y in train_pbar:
                batch_X, batch_y = batch_X.to(
                    self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)

                # Prepare targets for loss calculation
                targets = self._prepare_targets_for_loss(batch_y)
                loss = criterion(outputs.squeeze() if self.is_binary else outputs,
                                 targets.squeeze() if self.is_binary else targets)

                # Add L2 regularization
                if self.use_multi_gpu:
                    loss += model.module.get_penalty_loss()
                else:
                    loss += model.get_penalty_loss()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_pbar.close()

            # Evaluation
            val_score = self._evaluate_model(
                model, self.test_source, self.test_target)

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss/len(train_loader):.4f}',
                'val_score': f'{val_score:.4f}',
                'best': f'{best_val_score:.4f}'
            })

            # Keep track of best model
            if val_score > best_val_score:
                best_val_score = val_score
                # Save model state
                if self.use_multi_gpu:
                    best_model_state = model.module.state_dict()
                else:
                    best_model_state = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Early stopping
            if no_improve_epochs >= self.patience:
                epoch_pbar.set_description(
                    f"Early stopping at epoch {epoch+1}")
                break

        epoch_pbar.close()

        # Load best model state
        if self.use_multi_gpu:
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

        return model, best_val_score

    def _evaluate_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Evaluate model performance.

        Args:
            model: PyTorch model
            X: Features
            y: Targets

        Returns:
            Evaluation score (F1 for binary, accuracy for multiclass)
        """
        model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            # Create progress bar for evaluation if dataset is large
            batch_size = 1000  # Evaluation batch size
            num_samples = X.shape[0]
            all_predictions = []

            eval_pbar = tqdm(range(0, num_samples, batch_size),
                             desc="Evaluating", leave=False)

            for start_idx in eval_pbar:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X[start_idx:end_idx]

                outputs = model(batch_X)

                if self.is_binary:
                    batch_predictions = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    batch_predictions = torch.argmax(outputs, dim=1)

                all_predictions.append(batch_predictions.cpu())

            eval_pbar.close()

            # Concatenate all predictions
            if self.is_binary:
                predictions = torch.cat(all_predictions, dim=0).numpy()
                targets = y.cpu().numpy()

                # Handle different target formats
                if targets.shape[1] == 1:
                    # Convert continuous targets to binary for evaluation
                    binary_targets = (targets > 0.5).astype(int)
                    binary_predictions = predictions.astype(int)
                    return f1_score(binary_targets.ravel(), binary_predictions.ravel())
                else:
                    # One-hot encoded
                    y_true = np.argmax(targets, axis=1)
                    y_pred = np.argmax(predictions, axis=1)
                    return f1_score(y_true, y_pred)
            else:
                predictions = torch.cat(all_predictions, dim=0).numpy()
                targets = self._prepare_targets_for_loss(y)

                targets = targets.cpu().numpy()

                return accuracy_score(targets, predictions)

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> 'TorchRProbabilitiesClassifier':
        """
        Train the model using grid search for hyperparameters.

        Args:
            X: Optional features array (if None, uses self.source)
            y: Optional targets array (if None, uses self.target)

        Returns:
            self: The fitted classifier
        """
        # For compatibility with scikit-learn API
        if X is not None and y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            self.source = torch.tensor(X, dtype=torch.float32)
            self.target = torch.tensor(y, dtype=torch.float32)
            self.is_binary, self.output_size, self.num_classes, self.class_mapping = self._analyze_target(
                y)

        # Create DataLoader for batch training
        dataset = TensorDataset(self.source, self.target)
        train_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        # Get loss function based on task type
        criterion = self._get_loss_function()

        # Grid search
        print(
            f"Starting grid search with {len(self.lr_options) * len(self.penalty_options)} parameter combinations")
        print(
            f"Task type: {'Binary' if self.is_binary else 'Multiclass'} classification")
        print(
            f"Number of classes: {self.num_classes}, Output size: {self.output_size}")
        if self.class_mapping is not None:
            print(f"Using class mapping due to missing labels")

        # Progress bar for hyperparameter combinations
        param_combinations = list(itertools.product(
            self.lr_options, self.penalty_options))
        param_pbar = tqdm(param_combinations, desc="Grid Search")

        for lr, penalty in param_pbar:
            param_pbar.set_description(
                f"Grid Search - lr={lr}, penalty={penalty}")

            # Build model
            model = self._build_model()

            # Setup optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train and evaluate model
            trained_model, val_score = self._train_evaluate_model(
                model, optimizer, criterion, train_loader, penalty
            )

            param_pbar.set_postfix(
                {'val_score': f'{val_score:.4f}', 'best': f'{self.best_score:.4f}'})

            # Update best model if this one is better
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_lr = lr
                self.best_penalty = penalty

                # Save the model
                if self.use_multi_gpu:
                    self.best_model = SimpleNet(
                        self.input_size, self.hidden_size, self.output_size)
                    self.best_model.load_state_dict(
                        trained_model.module.state_dict())
                else:
                    self.best_model = SimpleNet(
                        self.input_size, self.hidden_size, self.output_size)
                    self.best_model.load_state_dict(trained_model.state_dict())

                self.best_model = self.best_model.to(self.device)

        param_pbar.close()

        print(
            f"Best hyperparameters: lr={self.best_lr}, penalty={self.best_penalty}")
        print(f"Best validation score: {self.best_score:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the best model from grid search.

        Args:
            X: Optional features array (if None, uses self.test_source)

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first.")

        # For compatibility with scikit-learn API
        if X is not None:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = self.test_source.to(self.device)

        # Make predictions
        self.best_model.eval()

        # Handle large datasets with progress bar
        batch_size = 1000
        num_samples = X_tensor.shape[0]
        all_predictions = []

        predict_pbar = tqdm(range(0, num_samples, batch_size),
                            desc="Predicting", leave=False)

        with torch.no_grad():
            for start_idx in predict_pbar:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X_tensor[start_idx:end_idx]

                outputs = self.best_model(batch_X)

                if self.is_binary:
                    batch_predictions = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    batch_predictions = torch.argmax(
                        outputs, dim=1).unsqueeze(1)

                all_predictions.append(batch_predictions.cpu())

        predict_pbar.close()
        predictions = torch.cat(all_predictions, dim=0).numpy()

        # Map predictions back to original class indices if needed
        if not self.is_binary and self.class_mapping is not None:
            reverse_mapping = self.class_mapping['reverse']
            mapped_predictions = np.zeros_like(predictions)
            for i, pred in enumerate(predictions):
                mapped_predictions[i] = reverse_mapping[int(pred.item())]
            return mapped_predictions

        return predictions

    def predict_proba(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities using the best model from grid search.

        Args:
            X: Optional features array (if None, uses self.test_source)

        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first.")

        # For compatibility with scikit-learn API
        if X is not None:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = self.test_source.to(self.device)

        # Make probability predictions
        self.best_model.eval()

        # Handle large datasets with progress bar
        batch_size = 1000
        num_samples = X_tensor.shape[0]
        all_probs = []

        proba_pbar = tqdm(range(0, num_samples, batch_size),
                          desc="Predicting Probabilities", leave=False)

        with torch.no_grad():
            for start_idx in proba_pbar:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X_tensor[start_idx:end_idx]

                outputs = self.best_model(batch_X)

                if self.is_binary:
                    # Single sigmoid output for binary classification
                    batch_probs = torch.sigmoid(outputs)
                    # Return probabilities for both classes [1-p, p]
                    batch_probs = torch.cat(
                        [1 - batch_probs, batch_probs], dim=1)
                else:
                    # Multiclass probabilities
                    batch_probs = torch.softmax(outputs, dim=1)

                all_probs.append(batch_probs.cpu())

        proba_pbar.close()
        probs = torch.cat(all_probs, dim=0).numpy()

        # For multiclass with missing labels, we need to expand probabilities
        # to include zero probabilities for missing classes
        if not self.is_binary and self.class_mapping is not None:
            reverse_mapping = self.class_mapping['reverse']
            max_original_class = max(reverse_mapping.values())
            expanded_probs = np.zeros((probs.shape[0], max_original_class + 1))

            for new_idx, orig_class in reverse_mapping.items():
                expanded_probs[:, orig_class] = probs[:, new_idx]

            return expanded_probs

        return probs

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters found during grid search.

        Returns:
            Dictionary of best hyperparameters
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first.")

        return {
            'learning_rate': self.best_lr,
            'penalty': self.best_penalty,
            'validation_score': self.best_score,
            'is_binary': self.is_binary,
            'num_classes': self.num_classes,
            'output_size': self.output_size,
            'has_missing_classes': self.class_mapping is not None,
            'class_mapping': self.class_mapping
        }
