import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler




class UCEModelWithTemperature(nn.Module):
    """
    A decorator that applies Temperature Scaling for Uncertainty Calibration.
    Instead of scaling logits, this scales uncertainty scores directly.
    """

    def __init__(self, model, eval_values, eval_correctness, test_values, test_correctness):
        super(UCEModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temp value
        self.eval_values = eval_values
        #self.ece_eval_values = ece_eval_values
        self.eval_correctness = eval_correctness

        self.test_values = test_values
        self.test_correctness = test_correctness



    def forward(self, input):
        logits = self.model(input)
        return logits  # We don't scale logits, only uncertainty scores

    def temperature_scale(self, uncertainties):
        """
        Apply temperature scaling to uncertainty scores
        """
        #return uncertainties / self.temperature  # Scale uncertainty values directly
        eps = 1e-10
        scaled = uncertainties / self.temperature.clamp(min=eps, max=100)
        return scaled.clamp(eps, 1 - eps)

    def set_temperature(self ):
        """
        Tune the temperature to optimize Uncertainty Calibration Error (UCE).
        """

        self.cuda()
        uce_criterion = _UCELoss().cuda()
        ece_criterion = _ECELoss().cuda()
        nll_criterion = nn.BCELoss().cuda()

        # Collect all the uncertainty scores and correctness labels
        

        

        # Compute initial UCE before scaling
        print(self.eval_values, "------")
        before_temperature_uce = nll_criterion(self.eval_values, 1-self.eval_correctness).item()
        #before_ece = ece_criterion(self.ece_eval_values, self.eval_correctness).item()
        print(f"Before Temperature - UCE: {before_temperature_uce:.4f}")

        # Optimize temperature parameter using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled_uncertainties = self.temperature_scale(self.eval_values)
            print(scaled_uncertainties)
            loss = nll_criterion(scaled_uncertainties, 1-self.eval_correctness)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Compute UCE after temperature scaling
        after_temperature_uce = nll_criterion(self.temperature_scale(self.eval_values), 1-self.eval_correctness).item()
        #after_ece = ece_criterion( self.temperature_scale(self.ece_eval_values), self.eval_correctness).item()
        print(f"Optimal Temperature: {self.temperature.item():.4f}")
        print(f"After Temperature - UCE: {after_temperature_uce:.4f}")

        return self

    def evaluate(self):
        """
        Evaluate uncertainty calibration on the test set.
        """
        self.cuda()
        uce_criterion = _UCELoss().cuda()
        ece_criterion = _ECELoss().cuda()

        self.model.eval()

        

        # Compute UCE before and after calibration
        before_temperature_uce = uce_criterion(self.test_values, 1-self.test_correctness).item()
        #draw_uce_reliability_graph(uncertainties, label, pred, num_bins=10, save_path="./saved_models", save_name="before_calibration")
        #draw_uce_reliability_graph(uncertainties, label, pred, num_bins=10, save_path="./saved_models", save_name="before_calibration")
        #auc_before = roc_auc_score(1-errors.cpu().detach().numpy(), uncertainties.cpu().numpy())
        after_temperature_uce = uce_criterion(self.temperature_scale(self.test_values),  1-self.test_correctness).item()
        #draw_uce_reliability_graph(self.temperature_scale(uncertainties), label,pred, num_bins=10, save_path="./saved_models", save_name="after_calibration")
        #draw_uce_reliability_graph(uncertainties, label, pred, num_bins=10, save_path="./saved_models", save_name="before_calibration")
        #auc_after = roc_auc_score(1-errors.cpu().detach().numpy(), self.temperature_scale(uncertainties).cpu().detach().numpy())


        print(f"Before Temperature - UCE : {before_temperature_uce:.4f}")
        print(f"After Temperature - UCE : {after_temperature_uce:.4f}")
        return self.temperature_scale(self.test_values)




class eECEModelWithTemperature(nn.Module):
    """
    A decorator that applies Temperature Scaling for Uncertainty Calibration.
    Instead of scaling logits, this scales uncertainty scores directly.
    """

    def __init__(self, model, eval_values, eval_correctness, test_values, test_correctness):
        super(ECEModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temp value
        self.eval_values = eval_values
        #self.ece_eval_values = ece_eval_values
        self.eval_correctness = eval_correctness

        self.test_values = test_values
        self.test_correctness = test_correctness



    def forward(self, input):
        logits = self.model(input)
        return logits  # We don't scale logits, only uncertainty scores

    def temperature_scale(self, uncertainties):
        """
        Apply temperature scaling to uncertainty scores
        """
        return uncertainties / self.temperature  # Scale uncertainty values directly

    def set_temperature(self ):
        """
        Tune the temperature to optimize Uncertainty Calibration Error (UCE).
        """

        self.cuda()
        #uce_criterion = _UCELoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # Collect all the uncertainty scores and correctness labels
        

        

        # Compute initial UCE before scaling
        before_temperature_uce = ece_criterion(self.eval_values, self.eval_correctness).item()
        #before_ece = ece_criterion(self.ece_eval_values, self.eval_correctness).item()
        print(f"Before Temperature - UCE: {before_temperature_uce:.4f}")

        # Optimize temperature parameter using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled_uncertainties = self.temperature_scale(self.eval_values)
            loss = ece_criterion(scaled_uncertainties, self.eval_correctness)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Compute UCE after temperature scaling
        after_temperature_uce = ece_criterion(self.temperature_scale(self.eval_values), self.eval_correctness).item()
        #after_ece = ece_criterion( self.temperature_scale(self.ece_eval_values), self.eval_correctness).item()
        print(f"Optimal Temperature: {self.temperature.item():.4f}")
        print(f"After Temperature - UCE: {after_temperature_uce:.4f}")

        return self

    def evaluate(self):
        """
        Evaluate uncertainty calibration on the test set.
        """
        self.cuda()
        #uce_criterion = _UCELoss().cuda()
        ece_criterion = _ECELoss().cuda()

        self.model.eval()

        

        # Compute UCE before and after calibration
        before_temperature_uce = ece_criterion(self.test_values, self.test_correctness).item()
        after_temperature_uce = ece_criterion(self.temperature_scale(self.test_values),  self.test_correctness).item()
        

        print(f"Before Temperature - UCE : {before_temperature_uce:.4f}")
        print(f"After Temperature - UCE : {after_temperature_uce:.4f}")
        return self.temperature_scale(self.test_values)





class _UCELoss(nn.Module):
    """
    Calculates Uncertainty Calibration Error (UCE).
    Bins uncertainty scores and compares them with actual error rates.
    """

    def __init__(self, n_bins=10):
        super(_UCELoss, self).__init__()
        self.n_bins = n_bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, uncertainties, errors):
        uce = torch.zeros(1, device=uncertainties.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (uncertainties.gt(bin_lower.item()) & uncertainties.le(bin_upper.item()))
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                error_rate_in_bin = errors[in_bin].mean()
                avg_uncertainty_in_bin = uncertainties[in_bin].mean()
                uce += torch.abs(avg_uncertainty_in_bin - error_rate_in_bin) * prop_in_bin

        return uce


class e_ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, correctness):
        #softmaxes = F.softmax(logits, dim=1)

        #confidences, predictions = torch.max(softmaxes, 1)

        accuracies = correctness.float() #preds.eq(labels)

        #acc = torch.sum(accuracies).item() / len(correctness)
        acc = correctness.float().mean().item()
        # print('Accuracy: %.4f' % acc)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = logits.gt(bin_lower.item()) * logits.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = logits[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece 



def train_UCE_temperature(args, model, eval_values, eval_correctness, test_values, test_correctness):
    
    # Now we're going to wrap the model with a decorator that adds temperature scaling
    uce_model = UCEModelWithTemperature(model, eval_values, eval_correctness, test_values, test_correctness)

    # Tune the model temperature, and save the results
    uce_model.set_temperature()
    probs = uce_model.evaluate()
    return probs


def etrain_ECE_temperature(args, model, eval_values, eval_correctness, test_values, test_correctness):
    
    # Now we're going to wrap the model with a decorator that adds temperature scaling
    ece_model = ECEModelWithTemperature(model, eval_values, eval_correctness, test_values, test_correctness)

    # Tune the model temperature, and save the results
    ece_model.set_temperature()
    probs = ece_model.evaluate()
    return probs


import torch
import torch.nn as nn
import torch.optim as optim

def train_ECE_temperature(args, model, eval_values, eval_correctness, test_values, test_correctness):
    """
    Trains a temperature scaling model to minimize NLL loss on confidence values (1 - uncertainty).
    """
    # Wrap the model with a temperature scaling decorator
    ece_model = ECEModelWithTemperature(model, eval_values, eval_correctness, test_values, test_correctness)

    # Optimize temperature scaling
    ece_model.set_temperature()
    
    # Evaluate the model with the learned temperature
    probs = ece_model.evaluate()
    return probs

class ECEModelWithTemperature(nn.Module):
    """
    A decorator that applies Temperature Scaling for Expected Calibration Error (ECE).
    This scales confidence values directly (which are 1 - uncertainty scores).
    """

    def __init__(self, model, eval_values, eval_correctness, test_values, test_correctness):
        super(ECEModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temperature value
        self.eval_values = eval_values  # Confidence values (1 - uncertainty)
        self.eval_correctness = eval_correctness  # Correctness labels (0/1)
        self.test_values = test_values  # Test confidence values (1 - uncertainty)
        self.test_correctness = test_correctness  # Test correctness labels (0/1)

    def forward(self, input):
        logits = self.model(input)
        return logits  # No need to scale logits, we scale confidence scores instead

    def temperature_scale(self, confidences):
        """
        Apply temperature scaling to confidence scores (which are already 1 - uncertainty).
        """
        #return confidences /self.temperature #** (1 / self.temperature)  # Scaling transformation

        eps = 1e-10
        scaled = confidences ** (1 / self.temperature) .clamp(min=eps, max=100)
        return scaled.clamp(eps, 1 - eps)

    def set_temperature(self):
        """
        Tune the temperature parameter to optimize NLL loss on confidence values.
        """

        self.cuda()
        nll_criterion = nn.BCELoss().cuda()  # Binary Cross-Entropy Loss for confidence calibration
        assert not torch.isnan(self.eval_values).any(), "NaN detected in logits!"
        print("Unique target values:", self.eval_correctness.unique())
        print("Min & Max of logits before sigmoid:", self.eval_values.min().item(), self.eval_values.max().item())
        # Compute initial NLL loss before scaling
        before_temperature_nll = nll_criterion(self.eval_values, self.eval_correctness).item()
        print(f"Before Temperature - NLL Loss: {before_temperature_nll:.4f}")

        # Optimize temperature parameter using LBFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled_confidences = self.temperature_scale(self.eval_values)
            loss = nll_criterion(scaled_confidences, self.eval_correctness)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Compute NLL loss after temperature scaling
        scaled = self.temperature_scale(self.eval_values)

        print("scaled min:", scaled.min().item(), "max:", scaled.max().item())
        print("scaled finite:", torch.isfinite(scaled).all().item())
        print("targets unique:", self.eval_correctness.unique().tolist())
        print("targets dtype:", self.eval_correctness.dtype)
        after_temperature_nll = nll_criterion(self.temperature_scale(self.eval_values), self.eval_correctness).item()
        print(f"Optimal Temperature: {self.temperature.item():.4f}")
        print(f"After Temperature - NLL Loss: {after_temperature_nll:.4f}")

        return self

    def evaluate(self):
        """
        Evaluate ECE on the test set after applying temperature scaling.
        """
        self.cuda()
        ece_criterion = _ECELoss().cuda()  # ECE loss for evaluation

        self.model.eval()

        # Compute ECE before and after calibration
        before_temperature_ece = ece_criterion(self.test_values, self.test_correctness).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(self.test_values), self.test_correctness).item()

        print(f"Before Temperature - ECE: {before_temperature_ece:.4f}")
        print(f"After Temperature - ECE: {after_temperature_ece:.4f}")

        return self.temperature_scale(self.test_values)  # Return calibrated confidence scores

class _ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) loss.
    """

    def __init__(self, n_bins=10):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, correctness):
        """
        Compute ECE based on calibrated confidence scores.
        """
        accuracies = correctness.float()
        ece = torch.zeros(1, device=confidences.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item()))
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece




    





