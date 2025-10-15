import torch
import os
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

# Small value to avoid log(0) issues
EPS = 1e-6  

def get_model_path(metric_name, model_dir):
    """Generates a unique file path for each uncertainty metric inside the specified directory."""
    return os.path.join(model_dir, f"platt_scaling_{metric_name}.pth")

def train_platt_scaling(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir, num_epochs=500, lr=0.01, k=5):
    """
    Trains a Platt Scaling model using 5-Fold Cross-Validation and saves it in the given directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = get_model_path(metric_name, model_dir)

    # Detect the device from the input tensors
    device = uncertainty_scores_train.device

    # Ensure correctness labels are on the same device
    correctness_labels_train = correctness_labels_train.to(device)

    # Convert uncertainty scores to log-odds
    uncertainty_scores_train = uncertainty_scores_train.clamp(EPS, 1 - EPS)
    logits_train = torch.log(uncertainty_scores_train)  # log transformation

    # Initialize parameters on the same device
    alpha = torch.randn(1, requires_grad=True, device=device)
    beta = torch.randn(1, requires_grad=True, device=device)

    # Define optimizer
    optimizer = optim.Adam([alpha, beta], lr=lr)

    # Loss function: Binary Cross Entropy with logits
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Convert to numpy for cross-validation
    uncertainty_scores_np = logits_train
    correctness_labels_np = correctness_labels_train

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    calibrated_probs = torch.zeros_like(correctness_labels_np, dtype=torch.float32)

    for train_idx, val_idx in kf.split(uncertainty_scores_np):
        X_train, X_val = torch.tensor(uncertainty_scores_np[train_idx], device=device), torch.tensor(uncertainty_scores_np[val_idx], device=device)
        y_train, y_val = torch.tensor(correctness_labels_np[train_idx], device=device), torch.tensor(correctness_labels_np[val_idx], device=device)

        # Train the model on 4/5 of the data
        for epoch in range(num_epochs):
            logits_scaled = alpha * X_train + beta
            loss = loss_fn(logits_scaled, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Apply trained model to the 1/5 validation set
        logits_scaled_val = alpha * X_val + beta
        calibrated_probs[val_idx] = torch.sigmoid(logits_scaled_val)

    # Save optimized parameters
    torch.save({'alpha': alpha.item(), 'beta': beta.item()}, model_path)
    print(f"[{metric_name}] Model trained with 5-Fold CV and saved in {model_dir}: α = {alpha.item():.4f}, β = {beta.item():.4f}")

def apply_platt_scaling(uncertainty_scores_test, metric_name, model_dir):
    """
    Applies Platt scaling to test uncertainty scores using a trained model for a specific uncertainty metric.
    """
    model_path = get_model_path(metric_name, model_dir)

    # Detect the device from the input tensor
    device = uncertainty_scores_test.device

    # Check if trained model exists
    if os.path.exists(model_path):
        params = torch.load(model_path, map_location=device)  # Load to correct device
        alpha = torch.tensor(params['alpha'], device=device)
        beta = torch.tensor(params['beta'], device=device)
        print(f"[{metric_name}] Loaded trained Platt scaling model from {model_dir}.")
    else:
        raise FileNotFoundError(f"[{metric_name}] Platt scaling model not found in {model_dir}. Train the model first.")

    # Convert test uncertainty scores to log-odds
    uncertainty_scores_test = uncertainty_scores_test.clamp(EPS, 1 - EPS)
    logits_test = torch.log(uncertainty_scores_test)

    # Apply Platt scaling
    logits_scaled_test = alpha * logits_test + beta
    calibrated_probs_test = torch.sigmoid(logits_scaled_test)  # Convert to probabilities

    return calibrated_probs_test

def platt_scaling_pipeline(uncertainty_scores_train, correctness_labels_train, uncertainty_scores_test, metric_name, model_dir):
    """
    Full pipeline: Check if model exists for the given uncertainty metric in the specified directory, train if necessary, then apply Platt scaling.
    """
    model_path = get_model_path(metric_name, model_dir)

    # Ensure all tensors are on the same device as uncertainty_scores_train
    device = uncertainty_scores_train.device
    correctness_labels_train = correctness_labels_train.to(device)
    uncertainty_scores_test = uncertainty_scores_test.to(device)

    # Train if model doesn't exist
    if not os.path.exists(model_path):
        print(f"[{metric_name}] Training Platt scaling model using 5-Fold Cross-Validation...")
        train_platt_scaling(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir)

    # Apply calibration on test set
    calibrated_probs = apply_platt_scaling(uncertainty_scores_test, metric_name, model_dir)

    return calibrated_probs















def uce_train_platt_scaling(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir, num_epochs=500, lr=0.01, k=5):
    """
    Trains a Platt Scaling model using 5-Fold Cross-Validation and saves it in the given directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = get_model_path(metric_name, model_dir)

    # Detect the device from the input tensors
    device = uncertainty_scores_train.device

    # Ensure correctness labels are on the same device
    correctness_labels_train = correctness_labels_train.to(device)

    # Convert uncertainty scores to log-odds
    uncertainty_scores_train = uncertainty_scores_train.clamp(EPS, 1 - EPS)
    logits_train = torch.log(uncertainty_scores_train)  # log transformation

    # Initialize parameters on the same device
    alpha = torch.randn(1, requires_grad=True, device=device)
    beta = torch.randn(1, requires_grad=True, device=device)

    # Define optimizer
    optimizer = optim.Adam([alpha, beta], lr=lr)

    # Loss function: Binary Cross Entropy with logits
    loss_fn = torch.nn.MSELoss() #torch.nn.BCEWithLogitsLoss()

    # Convert to numpy for cross-validation
    uncertainty_scores_np = logits_train
    correctness_labels_np = correctness_labels_train

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    calibrated_probs = torch.zeros_like(correctness_labels_np, dtype=torch.float32)

    for train_idx, val_idx in kf.split(uncertainty_scores_np):
        X_train, X_val = torch.tensor(uncertainty_scores_np[train_idx], device=device), torch.tensor(uncertainty_scores_np[val_idx], device=device)
        y_train, y_val = torch.tensor(correctness_labels_np[train_idx], device=device), torch.tensor(correctness_labels_np[val_idx], device=device)

        # Train the model on 4/5 of the data
        for epoch in range(num_epochs):
            logits_scaled = alpha * X_train + beta
            loss = loss_fn(logits_scaled, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Apply trained model to the 1/5 validation set
        logits_scaled_val = alpha * X_val + beta
        calibrated_probs[val_idx] = torch.sigmoid(logits_scaled_val)

    # Save optimized parameters
    torch.save({'alpha': alpha.item(), 'beta': beta.item()}, model_path)
    print(f"[{metric_name}] Model trained with 5-Fold CV and saved in {model_dir}: α = {alpha.item():.4f}, β = {beta.item():.4f}")

def uce_apply_platt_scaling(uncertainty_scores_test, metric_name, model_dir):
    """
    Applies Platt scaling to test uncertainty scores using a trained model for a specific uncertainty metric.
    """
    model_path = get_model_path(metric_name, model_dir)

    # Detect the device from the input tensor
    device = uncertainty_scores_test.device

    # Check if trained model exists
    if os.path.exists(model_path):
        params = torch.load(model_path, map_location=device)  # Load to correct device
        alpha = torch.tensor(params['alpha'], device=device)
        beta = torch.tensor(params['beta'], device=device)
        print(f"[{metric_name}] Loaded trained Platt scaling model from {model_dir}.")
    else:
        raise FileNotFoundError(f"[{metric_name}] Platt scaling model not found in {model_dir}. Train the model first.")

    # Convert test uncertainty scores to log-odds
    uncertainty_scores_test = uncertainty_scores_test.clamp(EPS, 1 - EPS)
    logits_test = torch.log(uncertainty_scores_test)

    # Apply Platt scaling
    logits_scaled_test = alpha * logits_test + beta
    calibrated_probs_test = torch.sigmoid(logits_scaled_test)  # Convert to probabilities

    return calibrated_probs_test

def uce_platt_scaling_pipeline(uncertainty_scores_train, correctness_labels_train, uncertainty_scores_test, metric_name, model_dir):
    """
    Full pipeline: Check if model exists for the given uncertainty metric in the specified directory, train if necessary, then apply Platt scaling.
    """
    model_path = get_model_path(metric_name, model_dir)

    # Ensure all tensors are on the same device as uncertainty_scores_train
    device = uncertainty_scores_train.device
    correctness_labels_train = correctness_labels_train.to(device)
    uncertainty_scores_test = uncertainty_scores_test.to(device)

    # Train if model doesn't exist
    if not os.path.exists(model_path):
        print(f"[{metric_name}] Training Platt scaling model using 5-Fold Cross-Validation...")
        uce_train_platt_scaling(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir)

    # Apply calibration on test set
    calibrated_probs = uce_apply_platt_scaling(uncertainty_scores_test, metric_name, model_dir)

    return calibrated_probs



import torch
import os
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np
#import ace_tools as tools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch.nn as nn


# Small value to avoid log(0) issues
EPS = 1e-6  

def newget_model_path(metric_name, model_dir):
    return os.path.join(model_dir, f"platt_scaling_{metric_name}.pth")

def newoptimize_separability(uncertainty_scores_train, correctness_labels_train, lr=0.01, epochs=500):
    """ Optimizes alpha and beta using PyTorch gradient descent to improve separability. """
    device = uncertainty_scores_train.device
    alpha = torch.tensor(1.2, requires_grad=True, device=device)
    beta = torch.tensor(0.8, requires_grad=True, device=device)
    optimizer = optim.Adam([alpha, beta], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        S_c = (uncertainty_scores_train[correctness_labels_train == 1] ** alpha).mean()
        S_i = (uncertainty_scores_train[correctness_labels_train == 0] ** beta).mean()
        loss = -S_c + S_i  # Maximize separation
        loss.backward()
        optimizer.step()
    
    return alpha.item(), beta.item()

def normalize_scores(uncertainty_scores):
    """
    Normalize uncertainty scores to improve separability before Platt scaling.
    """
    min_val = uncertainty_scores.min()
    max_val = uncertainty_scores.max()
    return (uncertainty_scores - min_val) / (max_val - min_val + 1e-6)


def compute_AUM(uncertainty_scores, correctness_labels):
    """
    Compute Area Under the Margin (AUM) for each sample.
    Higher AUM means the sample is easier to classify, lower AUM means it's harder.
    """
    device = uncertainty_scores.device

    correct_mask = correctness_labels == 1
    incorrect_mask = correctness_labels == 0

    # Compute mean uncertainty for correct & incorrect samples
    correct_mean = uncertainty_scores[correct_mask].mean() if torch.any(correct_mask) else torch.tensor(0.0, device=device)
    incorrect_mean = uncertainty_scores[incorrect_mask].mean() if torch.any(incorrect_mask) else torch.tensor(0.0, device=device)

    # Compute AUM as difference from class mean
    AUM_scores = torch.abs(uncertainty_scores - (correct_mean * correct_mask + incorrect_mean * incorrect_mask))

    return AUM_scores



def guided_mixup(uncertainty_scores, correctness_labels, AUM_scores, mixup_alpha=0.4):
    """
    Perform mixup based on AUM: higher AUM samples mix less, lower AUM samples mix more.
    """
    device = uncertainty_scores.device
    batch_size = uncertainty_scores.shape[0]

    # Shuffle indices for mixup pairing
    perm = torch.randperm(batch_size, device=device)

    # Compute mixup weights from AUM
    mixup_lambda = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample((batch_size,)).to(device)
    mixup_lambda = mixup_lambda * (1 - AUM_scores)  # Reduce mixup for high AUM

    # Interpolate uncertainty scores
    mixed_scores = mixup_lambda * uncertainty_scores + (1 - mixup_lambda) * uncertainty_scores[perm]

    # Interpolate correctness labels
    mixed_labels = mixup_lambda * correctness_labels + (1 - mixup_lambda) * correctness_labels[perm]

    return mixed_scores, mixed_labels

def train_guided_mixup_platt(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir, num_epochs=500, lr=0.01):
    """
    Train Platt Scaling with Guided Mixup.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"guided_mixup_platt_{metric_name}.pth")
    device = uncertainty_scores_train.device
    correctness_labels_train = correctness_labels_train.to(device)

    # Step 1: Compute AUM
    AUM_scores = compute_AUM(uncertainty_scores_train, correctness_labels_train)

    # Step 2: Apply Guided Mixup
    mixed_scores, mixed_labels = guided_mixup(uncertainty_scores_train, correctness_labels_train, AUM_scores)

    # Step 3: Convert to log-odds for Platt Scaling
    mixed_scores = mixed_scores.clamp(1e-6, 1 - 1e-6)
    logits_train = torch.log(mixed_scores)

    # Step 4: Train Platt Scaling
    alpha = torch.randn(1, requires_grad=True, device=device)
    beta = torch.randn(1, requires_grad=True, device=device)
    optimizer = optim.Adam([alpha, beta], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        logits_scaled = alpha * logits_train + beta
        loss = loss_fn(logits_scaled, mixed_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save trained parameters
    torch.save({'alpha': alpha.item(), 'beta': beta.item()}, model_path)
    print(f"[{metric_name}] Model trained with Guided Mixup and saved in {model_dir}")


def newapply_platt_scaling(uncertainty_scores_test, metric_name, model_dir):
    """
    Apply trained Guided Mixup + Platt Scaling to test uncertainty scores.
    """
    model_path = os.path.join(model_dir, f"guided_mixup_platt_{metric_name}.pth")
    device = uncertainty_scores_test.device

    if os.path.exists(model_path):
        # Load Platt scaling parameters
        params = torch.load(model_path, map_location=device)
        alpha = torch.tensor(params['alpha'], device=device)
        beta = torch.tensor(params['beta'], device=device)

        print(f"[{metric_name}] Loaded trained calibration model.")
    else:
        raise FileNotFoundError(f"[{metric_name}] Model not found. Train it first.")

    # Convert to log-odds
    uncertainty_scores_test = uncertainty_scores_test.clamp(1e-6, 1 - 1e-6)
    logits_test = torch.log(uncertainty_scores_test)

    # Apply Platt scaling
    logits_scaled_test = alpha * logits_test + beta
    calibrated_probs_test = torch.sigmoid(logits_scaled_test)

    return calibrated_probs_test


from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
    
def isotonic_regression(uncertainty_scores_train, correctness_labels_train, uncertainty_scores_test, metric_name, model_dir):
    
    """
    Pipeline to adjust test uncertainty scores.

    - Checks if a trained model exists.
    - If not found, trains a new model using training data.
    - Applies the trained model on test uncertainty scores.
    - Returns only the final adjusted scores.
    """

    """
    Pipeline to adjust test uncertainty scores using Isotonic Regression instead of GMM.

    - Checks if a trained model exists.
    - If not found, trains a new Isotonic Regression model using training data.
    - Applies the trained model on test uncertainty scores.
    - Returns only the final adjusted scores.
    """
    model_path = os.path.join(model_dir, f"{metric_name}_isotonic_adjustment.pth")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[{metric_name}] Model not found. Training a new Isotonic Regression model...")

        # Convert tensors to numpy arrays
        confidence_scores_train_np = uncertainty_scores_train.cpu().numpy()
        correctness_labels_train_np = correctness_labels_train.cpu().numpy()

        # Train Isotonic Regression model
        isotonic_reg = IsotonicRegression(out_of_bounds="clip")
        isotonic_reg.fit(confidence_scores_train_np, correctness_labels_train_np)

        # Save trained model
        torch.save({'isotonic_reg': isotonic_reg}, model_path)
        print(f"[{metric_name}] Isotonic Regression model trained and saved successfully.")

    else:
        print(f"[{metric_name}] Loaded trained Isotonic Regression model.")

    # Load trained model
    device = uncertainty_scores_test.device
    params = torch.load(model_path,weights_only=False,map_location=device)
    isotonic_reg = params['isotonic_reg']

    # Convert test scores to numpy
    confidence_scores_test_np = uncertainty_scores_test.cpu().numpy()

    # Apply Isotonic Regression transformation
    adjusted_scores_test = isotonic_reg.transform(confidence_scores_test_np)

    # Return only the final adjusted scores as a tensor
    return torch.tensor(adjusted_scores_test)



# Re-import necessary libraries after execution state reset
import os
import torch
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

def beta_calibration(uncertainty_scores_train, correctness_labels_train, uncertainty_scores_test, metric_name, model_dir):
    """
    Pipeline to adjust test uncertainty scores using Isotonic Regression + GMM Clustering.

    - Checks if a trained model exists.
    - If not found, trains a new Isotonic Regression model and Gaussian Mixture Model using training data.
    - Applies the trained model on test uncertainty scores.
    - Returns only the final adjusted scores.
    """
    model_path = os.path.join(model_dir, f"{metric_name}_isotonic_gmm_adjustment.pth")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[{metric_name}] Model not found. Training Isotonic Regression + GMM...")

        # Convert tensors to numpy arrays
        confidence_scores_train_np = uncertainty_scores_train.cpu().numpy()
        correctness_labels_train_np = correctness_labels_train.cpu().numpy()

        # Train Isotonic Regression model
        isotonic_reg = IsotonicRegression(out_of_bounds="clip")
        isotonic_reg.fit(confidence_scores_train_np, correctness_labels_train_np)

        # Apply Isotonic Regression to transform scores
        transformed_scores_train = isotonic_reg.transform(confidence_scores_train_np)

        # Train GMM on transformed scores
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(transformed_scores_train.reshape(-1, 1))

        # Save trained models
        torch.save({'isotonic_reg': isotonic_reg, 'gmm': gmm}, model_path)
        print(f"[{metric_name}] Isotonic Regression + GMM model trained and saved successfully.")

    else:
        print(f"[{metric_name}] Loaded trained Isotonic Regression + GMM model.")

    # Load trained model
    params = torch.load(model_path)
    isotonic_reg = params['isotonic_reg']
    gmm = params['gmm']

    # Convert test scores to numpy
    confidence_scores_test_np = uncertainty_scores_test.cpu().numpy()

    # Apply Isotonic Regression transformation
    transformed_scores_test = isotonic_reg.transform(confidence_scores_test_np)

    # Use GMM to predict class probabilities for the test set
    gmm_scores_test = gmm.predict_proba(transformed_scores_test.reshape(-1, 1))[:, 1]  # Probability of correct class

    # Return only the final adjusted scores as a tensor
    return torch.tensor(gmm_scores_test)
    
    
    
    
    
    
    
    
    
    
    
    
    """model_path = os.path.join(model_dir, f"{metric_name}_score_adjustment.pth")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[{metric_name}] Model not found. Training a new model...")

        # Convert tensors to numpy arrays
        confidence_scores_train_np = uncertainty_scores_train.cpu().numpy()
        correctness_labels_train_np = correctness_labels_train.cpu().numpy()

        # Compute empirical probability of correctness for each score
        unique_scores, counts = np.unique(confidence_scores_train_np, return_counts=True)
        correct_counts = np.array([np.sum(correctness_labels_train_np[confidence_scores_train_np == s]) for s in unique_scores])
        p_correct = correct_counts / counts
        p_correct = np.clip(p_correct, 1e-6, 1 - 1e-6)  # Avoid division errors

        # Compute probability ratio scores
        p_incorrect = 1 - p_correct
        probability_ratio_scores = p_correct / p_incorrect

        # Compute shifted scores
        lambda_factor = 0.5  # Tuning parameter
        adjusted_scores = unique_scores + lambda_factor * (p_correct - p_incorrect)

        # Fit Gaussian Mixture Model (GMM) for score transformation
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(confidence_scores_train_np.reshape(-1, 1))

        # Save trained parameters
        torch.save({
            'unique_scores': unique_scores,
            'adjusted_scores': adjusted_scores,
            'gmm': gmm
        }, model_path)

        print(f"[{metric_name}] Model trained and saved successfully.")

    else:
        print(f"[{metric_name}] Loaded trained model parameters.")

    # Load trained parameters
    params = torch.load(model_path)
    unique_scores = params['unique_scores']
    adjusted_scores = params['adjusted_scores']
    gmm = params['gmm']  # Gaussian Mixture Model

    # Convert test scores to numpy
    confidence_scores_test_np = uncertainty_scores_test.cpu().numpy()

    # Interpolate adjusted scores for test data
    adjusted_scores_test = np.interp(confidence_scores_test_np, unique_scores, adjusted_scores)

    # Use GMM to adjust scores probabilistically
    gmm_scores_test = gmm.predict_proba(confidence_scores_test_np.reshape(-1, 1))[:, 1]  # Probability of correct class

    # Combine both adjustments (weighted sum for final score)
    final_adjusted_scores = 0.5 * adjusted_scores_test + 0.5 * gmm_scores_test

    # Return only the final adjusted scores as a tensor
    return torch.tensor(final_adjusted_scores)"""
    
    """model_path = get_model_path(metric_name, model_dir)
    device = uncertainty_scores_train.device
    correctness_labels_train = correctness_labels_train.to(device)
    uncertainty_scores_test = uncertainty_scores_test.to(device)

    if not os.path.exists(model_path):
        print(f"[{metric_name}] Training model with separability enhancement...")
        train_guided_mixup_platt(uncertainty_scores_train, correctness_labels_train, metric_name, model_dir)

    calibrated_probs = newapply_platt_scaling(uncertainty_scores_test, metric_name, model_dir)
    return calibrated_probs.detach()"""

# Visualization of Separability
"""plt.figure(figsize=(8, 5))
plt.hist(uncertainty_scores_train[correctness_labels_train == 1].cpu().numpy(), bins=20, alpha=0.5, label='Correct - Original')
plt.hist(uncertainty_scores_train[correctness_labels_train == 0].cpu().numpy(), bins=20, alpha=0.5, label='Incorrect - Original')
plt.legend()
plt.title("Effect of Separability Transformation and Platt Scaling")
plt.show()

df_test_results = {'Original_Scores': uncertainty_scores_test.cpu().numpy(), 'Calibrated_Scores': calibrated_probs.cpu().numpy()}
tools.display_dataframe_to_user(name="Platt Scaled Test Scores", dataframe=pd.DataFrame(df_test_results))"""


