import torch

def calc_uce(uncertainties, labels, preds, bins=15):
    """
    Compute Uncertainty Calibration Error (UCE).
    
    Arguments:
    - uncertainties: Tensor of model uncertainty scores.
    - labels: Ground truth labels.
    - preds: Model predictions.
    - bins: Number of bins for reliability diagram.
    
    Returns:
    - UCE, bin boundaries, bin error rates, bin uncertainty values, bin sizes.
    """
    uncertainties = torch.tensor(uncertainties, dtype=torch.float32).clone()
    labels = torch.tensor(labels, dtype=torch.long).clone()
    preds = torch.tensor(preds, dtype=torch.long).clone()

    incorrect_preds = (preds != labels).float()  # 1 for incorrect, 0 for correct

    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    uce = torch.tensor(0.0,device=uncertainties.device)
    max_uce = torch.tensor(0.0, device=uncertainties.device)  # Initialize Max UCE
    bin_errors = torch.zeros(bins)
    bin_uncerts = torch.zeros(bins)
    bin_sizes = torch.zeros(bins)

    for i, bin_lower, bin_upper in zip(range(bins), bin_lowers, bin_uppers):
        in_bin = (uncertainties.gt(bin_lower.item()) & uncertainties.le(bin_upper.item()))
        prop_in_bin = in_bin.float().mean()
        bin_sizes[i] = prop_in_bin

        if prop_in_bin.item() > 0.0:
            error_rate_in_bin = incorrect_preds[in_bin].mean()
            avg_uncertainty_in_bin = uncertainties[in_bin].mean()
            bin_errors[i] = error_rate_in_bin
            bin_uncerts[i] = avg_uncertainty_in_bin

            #uce += torch.abs(avg_uncertainty_in_bin - error_rate_in_bin) * prop_in_bin

            abs_diff = torch.abs(avg_uncertainty_in_bin - error_rate_in_bin)
            uce += abs_diff * prop_in_bin
            max_uce = torch.max(max_uce, abs_diff)  # Track the maximum absolute difference

    print(f"UCE: {uce.item() * 100:.2f}%")

    #print(f"UCE: {uce.item() * 100:.2f}%")
    
    return uce.item(), max_uce.item(),bin_boundaries, bin_errors, bin_uncerts, bin_sizes


import torch
import torch.nn.functional as F

def calc_ece(logits, labels, preds, bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Arguments:
    - logits: Model output logits or probabilities.
    - labels: Ground truth labels.
    - preds: Model predictions.
    - bins: Number of bins for reliability diagram.
    
    Returns:
    - ECE, accuracy, bin boundaries, bin accuracies, bin confidences, bin sizes.
    """
    labels = torch.tensor(labels, dtype=torch.long).clone()
    preds = torch.tensor(preds, dtype=torch.long).clone()
    
    # Convert logits to confidence scores
    confidences = logits #torch.max(F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1), dim=1)[0]

    
    correctness = preds.eq(labels).float()
    print("accuracy: ", (preds == labels).sum().item() / len(labels))

    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0,device=logits.device)
    accuracy = torch.tensor(0.0,device=logits.device)
    max_ece = torch.tensor(0.0, device=logits.device)  # Max ECE initialization
    bin_accs = torch.zeros(bins)
    bin_confs = torch.zeros(bins)
    bin_sizes = torch.zeros(bins)

    for i, bin_lower, bin_upper in zip(range(bins), bin_lowers, bin_uppers):
        in_bin = logits.gt(bin_lower.item()) * logits.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        bin_sizes[i] = prop_in_bin

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = logits[in_bin].float().mean()
            bin_accs[i] = accuracy_in_bin
            bin_confs[i] = avg_confidence_in_bin
            
            abs_diff = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            #ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            max_ece = torch.max(max_ece, abs_diff)  # Track maximum absolute difference
            accuracy += accuracy_in_bin * prop_in_bin

    brier_score = torch.mean((confidences - correctness) ** 2).item()
    
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
    print(f"ECE: {ece.item() * 100:.2f}%")
    print(f"Brier Score: {brier_score:.4f}")

    return ece.item(), max_ece.item(), accuracy.item(), brier_score,bin_boundaries, bin_accs, bin_confs, bin_sizes


import os
import matplotlib.pyplot as plt

def draw_reliability_graph(logits, labels, preds, num_bins, save_path, save_name):
    """
    Draw the reliability diagram for Expected Calibration Error (ECE).
    """
    ECE, MAX_ECE, ACC, brier_score,bins, bin_accs, bin_confs, bin_sizes = calc_ece(logits, labels, preds, num_bins)
    bins = bins[1:]

    index_x = [b - (1.0 / (2 * num_bins)) for b in bins]  # Adjust x-position

    
    

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(index_x, bin_sizes, width=0.1, alpha=1, edgecolor='black', color='b')
    for i, j in zip(index_x, bin_sizes):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=16)
    plt.tick_params(labelsize=20)
    os.makedirs(save_path, exist_ok=True )
    plt.savefig(os.path.join(save_path, '{}-ecefrac.png'.format(save_name)), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color='gray', linestyle='dashed')

    plt.bar(index_x, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b', label='Output')
    gaps = [a - b for a, b in zip(bin_confs, bin_accs)]
    plt.bar(index_x, gaps, bottom=bin_accs, width=0.1, alpha=0.3, edgecolor='r', color='r', hatch='\\', label='Gap')

    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    plt.legend(fontsize=21)

    textstr = f"ECE = {ECE * 100:.2f}%\n"
    props = dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='black')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=25, verticalalignment='bottom',
            horizontalalignment='right', bbox=props)

    plt.tick_params(labelsize=20)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}-acc.png"), bbox_inches='tight')

    return ECE, MAX_ECE,ACC, brier_score


def draw_uce_reliability_graph(uncertainties, labels, preds, num_bins, save_path, save_name):
    """
    Draw the reliability diagram for Uncertainty Calibration Error (UCE).
    """
    UCE, MAX_UCE, bins, bin_errors, bin_uncerts, bin_sizes = calc_uce(uncertainties, labels, preds, num_bins)
    bins = bins[1:]

    index_x = [b - (1.0 / (2 * num_bins)) for b in bins]


    
    

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(index_x, bin_sizes, width=0.1, alpha=1, edgecolor='black', color='b')
    for i, j in zip(index_x, bin_sizes):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=16)
    plt.tick_params(labelsize=20)
    os.makedirs(save_path, exist_ok=True )
    plt.savefig(os.path.join(save_path, '{}-ucefrac.png'.format(save_name)), bbox_inches='tight')


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color='gray', linestyle='dashed')


    plt.bar(index_x, bin_errors, width=0.1, alpha=1, edgecolor='black', color='b', label='Output')
    gaps = [a - b for a, b in zip(bin_uncerts, bin_errors)]
    plt.bar(index_x, gaps, bottom=bin_errors, width=0.1, alpha=0.3, edgecolor='r', color='r', hatch='\\', label='Gap')

    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    plt.legend(fontsize=21)

    textstr = f"UCE = {UCE * 100:.2f}%"
    props = dict(boxstyle='round', alpha=0.5, facecolor='white', edgecolor='black')
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=25, verticalalignment='bottom',
            horizontalalignment='right', bbox=props)

    plt.tick_params(labelsize=20)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{save_name}-uncertainty.png"), bbox_inches='tight')

    return UCE, MAX_UCE
