from __future__ import absolute_import, division, print_function

import argparse
from cgi import test
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


from plots import draw_uce_reliability_graph, draw_reliability_graph
from sklearn.metrics import roc_auc_score
from emsemble import *
from scaling import *
from platt_scaling import *

from torch import nn, optim
from torch.nn import functional as F



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['input'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['id'],js['label'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def compute_vanilla(probabilities):
    uncertainties, _= torch.max(probabilities,dim=1)

    return uncertainties


def compute_entropy(probabilities):
    #pred_entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    uncertainties = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)


    bald_min = uncertainties.min()
    bald_max = uncertainties.max()
    uncertainties = (uncertainties - bald_min) / (bald_max - bald_min + 1e-8)

    return uncertainties

def compute_mutual_information(probabilities):
    """Compute Mutual Information for a single deterministic softmax output."""
    eps = 1e-10  # Small value to prevent log(0)
    num_classes = probabilities.shape[1]
    
    # Compute entropy of the output distribution
    entropy_term = -torch.sum(probabilities * torch.log(probabilities + eps), dim=1)
    
    # Compute marginal entropy (assuming uniform distribution over classes)
    marginal_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    
    # Mutual Information = Marginal Entropy - Expected Entropy
    uncertainties = marginal_entropy - entropy_term

    bald_min = uncertainties.min()
    bald_max = uncertainties.max()
    uncertainties = (uncertainties - bald_min) / (bald_max - bald_min + 1e-8)
    return uncertainties


def margin_confidence(probabilities):
    if probabilities.shape[1] < 2:
        raise ValueError("Margin calculation requires at least two classes.")
    sorted_scores, _ = torch.sort(probabilities, dim=1, descending=True)
    uncertainties = sorted_scores[:, 0] - sorted_scores[:, 1]
    return uncertainties

def least_confidence(probabilities):
    uncertainties =torch.min(probabilities, dim=1)[0]  # Lowest probability in the distribution
    return uncertainties

def ratio_confidence(probabilities):
    sorted_scores, _ = torch.sort(probabilities, dim=1, descending=True)
    uncertainties = sorted_scores[:, 1] / sorted_scores[:, 0]
    return uncertainties

def best_vs_second_best(probabilities):
    sorted_scores, _ = torch.sort(probabilities, dim=1, descending=True)
    uncertainties = sorted_scores[:, 0] - sorted_scores[:, 1]
    return uncertainties

def activate_dropout(model):
        """Enable dropout layers at inference time."""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()  # Keep dropout active

"""def monte_carlo_dropout(model, dataloader, num_samples=100):
        #Perform MC Dropout by running multiple stochastic forward passes.
        model.eval()  # Set model to evaluation mode
        activate_dropout(model)  # Ensure dropout remains active

        prediction_probs = []
        with torch.no_grad():
            for _ in range(num_samples):
                batch_probs = []
                for batch in tqdm(dataloader, desc=f"Evaluating MCD Sample {_+1}/{num_samples}"):
                    inputs = batch[0].cuda()
                    logits = model.forward_with_logits(inputs)
                    probs = F.softmax(logits, dim=-1)  # Convert to probabilities
                    batch_probs.append(probs)
                prediction_probs.append(torch.cat(batch_probs))
                print(prediction_probs)

        predictions = torch.stack(prediction_probs)  # Shape: (num_samples, batch_size, num_classes)
        mean_prediction = predictions.mean(dim=0)  # Expected output
        uncertainty = predictions.std(dim=0)  # Standard deviation as uncertainty

        return mean_prediction, uncertainty"""

def monte_carlo_dropout_logits(model, dataloader, num_samples=100):
    """Perform MC Dropout, collecting logits over multiple stochastic passes."""
    model.eval()
    activate_dropout(model)  # Keep dropout active at inference

    prediction_logits = []  # Stores logits instead of probabilities
    with torch.no_grad():
        for _ in range(num_samples):
            batch_logits = []
            for batch in tqdm(dataloader, desc=f"Evaluating MCD Sample {_+1}/{num_samples}"):
                inputs = batch[0].cuda()
                logits = model.forward_with_logits(inputs)  # Get raw logits
                batch_logits.append(logits)
            prediction_logits.append(torch.cat(batch_logits))

    return torch.stack(prediction_logits)  # Shape: (num_samples, batch_size, num_classes)


def compute_sws(logits):
    """
    Compute Sampled Winning Score (SWS) as the average of the max probability values.

    Args:
        logits: Tensor of shape (iter_time, batch_size, num_classes), MC Dropout logits.
        iter_time: Number of MC Dropout samples.

    Returns:
        sws_mean: Mean max probability per sample (batch_size,).
    """
    mc_probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    max_probs, _ = mc_probs.max(dim=-1)  # Take the highest probability per sample

    # Compute the mean of the max probabilities over all MC Dropout samples
    sws_mean = max_probs.mean(dim=0)

    return sws_mean  # Higher value = more confidence


def compute_pv(logits):
    """
    Compute Probability Variance (PV) using variance of softmax probabilities.

    Args:
        logits: Tensor of shape (iter_time, batch_size, num_classes), MC Dropout logits.

    Returns:
        pv_scores: Probability variance score (batch_size,).
    """
    mc_probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    pv_scores = mc_probs.var(dim=0).mean(dim=1)  # Variance over MC samples, mean over classes

    return pv_scores  # Higher variance = more uncertainty

def compute_bald(logits: torch.Tensor):
    """
    Compute BALD (Bayesian Active Learning by Disagreement) Score using PyTorch.

    Args:
        logits: Tensor of shape (iter_time, batch_size, num_classes), MC Dropout logits.

    Returns:
        bald_scores: BALD uncertainty scores (batch_size,).
    """
    mc_probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

    # Compute entropy of the mean probability distribution
    mean_probs = mc_probs.mean(dim=0)  # Mean probability over MC samples (batch_size, num_classes)
    mean_prob_uncertainties = (-mean_probs * mean_probs.log()).sum(dim=-1)  # Entropy of mean probabilities

    # Compute mean of entropies across MC samples
    sample_entropies = (-mc_probs * mc_probs.log()).sum(dim=-1)  # Entropy for each MC sample (iter_time, batch_size)
    mean_sample_entropy = sample_entropies.mean(dim=0)  # Mean entropy across MC samples (batch_size,)

    bald_score = mean_sample_entropy + mean_prob_uncertainties  # BALD score # Higher = more epistemic uncertainty
    #bald_score_normalized = bald_score / torch.log(torch.tensor(2, dtype=torch.float32))
    bald_min = bald_score.min()
    bald_max = bald_score.max()
    bald_score_normalized = (bald_score - bald_min) / (bald_max - bald_min + 1e-8)  # Small epsilon to prevent division by zero

    return bald_score_normalized

    #return bald_score_clamped


def train_deep_ensemble(args, model_class, config, tokenizer, num_models=5):
    """
    Train multiple independent models with different seeds for Deep Ensembles.

    Args:
        args: Argument parser containing training configurations.
        model_class: Model architecture (e.g., RoBERTa, BERT).
        config: Model configuration.
        tokenizer: Tokenizer for the model.
        num_models: Number of models to train.

    Returns:
        trained_model_paths: List of trained model paths.
    """
    base_dir = os.path.join( "uncertainty", "ensemble")
    os.makedirs(base_dir, exist_ok=True)
    
    trained_model_paths = []
    for i in range(num_models):
        seed = args.seed + i  # Different seed for each model
        set_seed(seed)
        model_dir = os.path.join(base_dir, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)
        args.output_dir = model_dir  # Unique folder for each model

        print(f"\nTraining Model {i+1}/{num_models} with seed {seed}...\n")
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model = Model(model, config, tokenizer, args)
        model.to(args.device)

        # Train model only when training is needed
        #train_model(model, tokenizer,args)

        trained_model_paths.append(model_dir)

    return trained_model_paths


def train_model(model, tokenizer,args):
    """
    Trains a model using given training arguments.

    Args:
        model: The model to be trained.
        args: Arguments containing training configurations.

    Returns:
        None (Saves trained model to `args.output_dir`).
    """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs= 2 #args.epoch
    model.to(args.device)
    
    """train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)"""

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    
    model.zero_grad()
    model.train()
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_dataloader):
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            loss,outputs = model(inputs, labels)
            #loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed.")

    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def load_ensemble_models(model_class, config, tokenizer, model_paths, args):
    """
    Load multiple trained models from different file paths.

    Args:
        model_class: The model architecture (e.g., RoBERTa, BERT).
        config: Model configuration.
        tokenizer: Tokenizer for the model.
        model_paths: List of trained model directories.

    Returns:
        models: List of loaded models.
    """
    models = []
    for path in model_paths:
        model = model_class(config)
        model = Model(model, config, tokenizer, args)  # Reinitialize model
        model.to("cuda")

        # Define model file path
        model_path = os.path.join(path, "model.pth")

        # Ensure the model file exists before loading
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")

        models.append(model)

    return models


def deep_ensemble_predictions(models, dataloader):
    """
    Perform inference with an ensemble of models and collect predictions.

    Args:
        models: List of trained models.
        dataloader: DataLoader with input samples.

    Returns:
        ensemble_logits: Tensor of shape (num_models, batch_size, num_classes).
    """
    num_models = len(models)
    ensemble_logits = []

    for model in models:
        batch_logits = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = batch[0].cuda()
                logits = model.forward_with_logits(inputs)  # Get raw logits
                batch_logits.append(logits)
        ensemble_logits.append(torch.cat(batch_logits))

    return torch.stack(ensemble_logits)  # Shape: (num_models, batch_size, num_classes)







def get_predictions(model, dataloader, args):
        
        prediction_probs = []
        labels_list = []
        pred_list = []

        model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = batch[0].cuda()
                labels = batch[1].cuda()
                logits = model.forward_with_logits(inputs)

                probs = F.softmax(logits, dim=-1)
               
                preds = torch.argmax(probs, dim=1)
                
                

                prediction_probs.append(probs)
                
                labels_list.append(labels)
                pred_list.append(preds)


        predictions = torch.cat(prediction_probs).cuda() 
        label = torch.cat(labels_list).cuda()
        pred = torch.cat(pred_list).cuda()

        return predictions, label, pred




def compute_accuracy(test_mc_logits, test_preds, test_probs):
    mc_probs = F.softmax(test_mc_logits, dim=-1)  # Shape: [num_inferences, num_samples, num_classes]

    # Get predicted classes for each MC inference
    mc_preds = mc_probs.argmax(dim=-1)  # Shape: [num_inferences, num_samples]

    mc_correctness = (mc_preds == test_preds).float()
    max_probs, _ = mc_probs.max(dim=-1)  # Max probability per inference

# Get top two probabilities per inference
    """top2_probs, _ = mc_probs.topk(2, dim=-1)  # Shape: [num_inferences, num_samples, 2]
    prob_diff = top2_probs[:, :, 0] - top2_probs[:, :, 1]

    num_correct_per_inference = mc_correctness.sum(dim=-1)  # Sum correctness over samples
    correct_scores = max_probs + prob_diff  # Score if correct
    incorrect_scores = max_probs - prob_diff  # Score if incorrect

    # Assign scores based on correctness
    mc_scores = torch.where(mc_correctness.bool(), correct_scores, incorrect_scores)  # Shape: [num_inferences, num_samples]

    # Take the average per sample across all inferences
    final_scores = mc_scores.mean(dim=0)  # Averaging across inferences (shape: [num_samples])
    print(final_scores.shape)"""

    #test_probs = test_probs.argmax(dim=-1)  # Original model predictions
    test_probs = test_probs.max(dim=-1)[0] 
    agreement = (mc_preds == test_preds.unsqueeze(0)).float()  # Shape: [num_inferences, num_samples]
    agreement_ratio = agreement.mean(dim=0)
    mc_prob_variance = max_probs.var(dim=0)  # Shape: [num_samples]

    mc_entropy = -(mc_probs * mc_probs.log()).sum(dim=-1).mean(dim=0)

    # Compute confidence-weighted alignment score
    #scores = agreement_ratio * (test_probs + mc_prob_variance)
    scores = agreement_ratio * (test_probs + mc_prob_variance -  mc_entropy)

# Convert to expected output shape torch.tensor([num_samples])
    final_scores = scores.view(-1)  # Shape: [num_samples]


    bald_min = final_scores.min()
    bald_max = final_scores.max()
    bald_score_normalized = (final_scores - bald_min) / (bald_max - bald_min + 1e-8)

    # Ensure output shape matches (torch.tensor([num_samples]))
    """print("Final aggregated scores shape:", final_scores.shape) 
    
    
    
    total_samples = test_preds.shape[0]

    print("Correctness per inference:")
    for i, correct_count in enumerate(num_correct_per_inference):
        print(f"Inference {i+1}: {correct_count}/{total_samples} correct ({(correct_count/total_samples)*100:.2f}%)")"""

    return bald_score_normalized


# Define Uncertainty Metrics

def save_uncertainty_scores(method, eval_scores, test_scores, base_dir="uncertainty"):
    """Saves uncertainty scores for eval and test sets under method-specific directory."""
    method_dir = os.path.join(base_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    eval_path = os.path.join(method_dir, "eval_scores.pt")
    test_path = os.path.join(method_dir, "test_scores.pt")
    
    torch.save(eval_scores, eval_path)
    torch.save(test_scores, test_path)

def load_uncertainty_scores(method, base_dir="uncertainty"):
    """Loads uncertainty scores if they exist."""
    method_dir = os.path.join(base_dir, method)
    eval_path = os.path.join(method_dir, "eval_scores.pt")
    test_path = os.path.join(method_dir, "test_scores.pt")
    
    if os.path.exists(eval_path) and os.path.exists(test_path):
        return torch.load(eval_path), torch.load(test_path)
    return None, None

def save_samples(method, mc_samples, ind = "path", base_dir="uncertainty"):
    """Saves Monte Carlo Dropout samples."""
    method_dir = os.path.join(base_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    if ind == "eval":
        mc_path = os.path.join(method_dir, "eval_mcd_samples.pt")
        torch.save(mc_samples, mc_path)
    else: 
        mc_path = os.path.join(method_dir, "test_mcd_samples.pt")
        torch.save(mc_samples, mc_path)

def load_samples(method, ind = "path", base_dir="uncertainty" ):
    """Loads Monte Carlo Dropout samples if they exist."""
    method_dir = os.path.join(base_dir, method)
    if ind == "eval":
        mc_path = os.path.join(method_dir, "eval_mcd_samples.pt")
    else:
        mc_path = os.path.join(method_dir, "test_mcd_samples.pt")
    
    if os.path.exists(mc_path):
        return torch.load(mc_path)
    return None






def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_loss_delta", type=float, default=0.001,
                        help="Minimum change in the loss required to qualify as an improvement.")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')


    

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    #device = torch.device("cuda:1")
    #torch.cuda.set_device(1)
    #device = torch.device("cuda:1")
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=4
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)


    checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir))#,map_location=device))      
    model.to(args.device)


    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)


    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)

    #torch.Size([2732])
    #torch.Size([2732])
    #torch.Size([2732, 2])

    eval_probs, eval_label, eval_preds = get_predictions(model, eval_dataloader, args)
    eval_correctness = (eval_preds == eval_label).float()  # 1 for correct, 0 for incorrect
    print("Unique target values:", eval_correctness.unique())
    
    eval_erroneousness = 1 - eval_correctness  # Complement of correctness
    test_probs, test_label, test_preds = get_predictions(model, test_dataloader, args)
    test_correctness = (test_preds == test_label).float()  # 1 for correct, 0 for incorrect
    test_erroneousness = 1 - test_correctness  # Complement of correctness

    device = test_label.device

    ROOT_DIR = "uncertainty"
    uncertainty_methods = ["vanilla", "entropy","mutual", "margin", "least", "ratio", "best", "mcd","ensemble", "dsmg" ] #["mcd","ensemble", "dsmg"]

    for method in uncertainty_methods:
        
        if method == "vanilla":
            eval_vanilla = compute_vanilla(eval_probs)
            test_vanilla = compute_vanilla(test_probs)
            
            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((1-test_vanilla).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE, MAX_UCE = draw_uce_reliability_graph(test_vanilla.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_vanilla, eval_correctness, test_vanilla, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_vanilla, eval_correctness, 1-test_vanilla, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_vanilla, eval_correctness, 1-test_vanilla, "vanilla",method_dir)
            ECE, ACC, MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_vanilla, 1-eval_correctness, test_vanilla, "new_uncertain_vanilla",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            



                


        elif method == "entropy":
            eval_entropy = compute_entropy(eval_probs)
            test_entropy = compute_entropy(test_probs)
            test_entropy1 = test_entropy.cpu().numpy()

            with open(os.path.join("./saved_models/","before_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,test_entropy1):
                    f.write(example.idx+'\t'+str(sc)+'\n')

            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC, MAX_ECE, brier_score = draw_reliability_graph((1-test_entropy).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_entropy.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling
            assert not torch.isnan(eval_entropy).any(), "NaN detected in logits!"
            print("skipped")
            uce_scaled_values = train_UCE_temperature(args, model, eval_entropy, eval_correctness, test_entropy, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_entropy, eval_correctness, 1-test_entropy, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            """uce_scaled_values1 = uce_scaled_values.cpu().numpy()

            with open(os.path.join("./saved_models/","after_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,uce_scaled_values1):
                    f.write(example.idx+'\t'+str(sc)+'\n')"""

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            
            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_entropy, eval_correctness, 1-test_entropy, "entropy",method_dir)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_entropy, 1-eval_correctness, test_entropy, "newuncertain_entropy",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            uce_scaled_values1 = uncertain_plat_scaled.cpu().numpy()

            with open(os.path.join("./saved_models/","platt_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,uce_scaled_values1):
                    f.write(example.idx+'\t'+str(sc)+'\n')


        elif method =="mutual":
            eval_mutual = compute_mutual_information(eval_probs)
            test_mutual = compute_mutual_information(test_probs)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_mutual).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_mutual.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_mutual, eval_correctness, test_mutual, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_mutual, eval_correctness, 1-test_mutual, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_mutual, eval_correctness, 1-test_mutual, "mutual",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_mutual, 1-eval_correctness, test_mutual, "new_uncertain_mutual",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ") 


        elif method =="margin":
            eval_margin = margin_confidence(eval_probs)
            test_margin = margin_confidence(test_probs)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_margin).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_margin.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_margin, eval_correctness, test_margin, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_margin, eval_correctness, 1-test_margin, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_margin, eval_correctness, 1-test_margin, "margin",method_dir)
            ECE, ACC, MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_margin, 1-eval_correctness, test_margin, "newuncertain_margin",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")



        elif method =="least":
            eval_least = least_confidence(eval_probs)
            test_least = least_confidence(test_probs)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_least).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_least.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_least, eval_correctness, test_least, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_least, eval_correctness, 1-test_least, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ") 


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_least, eval_correctness, 1-test_least, "least",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_least, 1-eval_correctness, test_least, "newuncertain_least",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ") 

        elif method =="ratio":
            eval_ratio = ratio_confidence(eval_probs)
            test_ratio = ratio_confidence(test_probs)

            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_ratio).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_ratio.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_ratio, eval_correctness, test_ratio, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_ratio, eval_correctness, 1-test_ratio, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_ratio, eval_correctness, 1-test_ratio, "ratio",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_ratio, 1-eval_correctness, test_ratio, "newuncertain_ratio",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


        elif method =="best":
            eval_best = best_vs_second_best(eval_probs)
            test_best = best_vs_second_best(test_probs)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_best).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_best.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_best, eval_correctness, test_best, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_best, eval_correctness, 1-test_best, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ") 

            
            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_best, eval_correctness, 1-test_best, "best",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_best, 1-eval_correctness, test_best, "newuncertain_best",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")




        elif method == "mcd":
            #eval_mc_logits = monte_carlo_dropout_logits(model, eval_dataloader, num_samples=5)
            #save_mcd_samples("mcd", eval_mc_logits)

            eval_mc_logits = load_samples(method, ind = "eval")  # Load MCD samples if they exist

            if eval_mc_logits is None:
                eval_mc_logits = monte_carlo_dropout_logits(model, eval_dataloader, num_samples=50)
                save_samples(method, eval_mc_logits, ind = "eval")  # Save MCD samples for future use
            
            eval_sws = compute_sws(eval_mc_logits)
            eval_pv = compute_pv(eval_mc_logits)
            eval_bald = compute_bald(eval_mc_logits)

            test_mc_logits = load_samples(method, ind = "test")  # Load MCD samples if they exist

            if test_mc_logits is None:
                test_mc_logits = monte_carlo_dropout_logits(model, test_dataloader, num_samples=50)
                save_samples(method, test_mc_logits, ind = "test")


            test_sws = compute_sws(test_mc_logits)
            test_pv = compute_pv(test_mc_logits)
            test_bald = compute_bald(test_mc_logits)
            """dropout variational like codeimprove """
            """results = compute_accuracy(test_mc_logits, test_preds, test_probs)

            uce_scaled_values1 = results.cpu().numpy()

            with open(os.path.join("./saved_models/","after_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,uce_scaled_values1):
                    f.write(example.idx+'\t'+str(sc)+'\n')"""


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph(test_sws.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_sws.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_before_scale" )

            scores_path = os.path.join(method_dir, "sws_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_sws, eval_correctness, test_sws, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, eval_sws, eval_correctness, test_sws, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_after_scale" )

            scores_path = os.path.join(method_dir, "sws_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(eval_sws, eval_correctness, test_sws, "sws",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_sws, 1-eval_correctness, test_sws, "newuncertain_sws",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "sws_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #pv

            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_pv).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_pv.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_before_scale" )

            scores_path = os.path.join(method_dir, "pv_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_pv, eval_correctness, test_pv, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model,1- eval_pv, eval_correctness, 1-test_pv, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_after_scale" )

            scores_path = os.path.join(method_dir, "pv_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_pv, eval_correctness, 1-test_pv, "pv",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_pv, 1-eval_correctness, test_pv, "newuncertain_pv",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "pv_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            #bald

            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((1-test_bald).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_bald.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_before_scale" )

            scores_path = os.path.join(method_dir, "bald_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_bald, eval_correctness, test_bald, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_bald, eval_correctness, 1-test_bald, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_after_scale" )

            scores_path = os.path.join(method_dir, "bald_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_bald, eval_correctness, 1-test_bald, "bald",method_dir)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_bald, 1-eval_correctness, test_bald, "newuncertain_bald",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "bald_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")









        elif method =="ensemble":
            #only when training requered

            eval_ensemble_logits = load_samples(method, ind = "eval")  # Load MCD samples if they exist

            if eval_ensemble_logits is None:
                trained_model_paths = train_deep_ensemble(args, model_class, config, tokenizer, num_models=5)
                models = load_ensemble_models(model_class, config, tokenizer, trained_model_paths, args)
                # Get Deep Ensemble logit samples
                eval_ensemble_logits = deep_ensemble_predictions(models, eval_dataloader)  # Shape: (num_models, batch_size, num_classes)
                save_samples(method, eval_ensemble_logits, ind = "eval")  # Save MCD samples for future use
            eval_sws = compute_sws(eval_ensemble_logits)
            eval_pv = compute_pv(eval_ensemble_logits)
            eval_bald = compute_bald(eval_ensemble_logits)


            test_ensemble_logits = load_samples(method, ind = "test")  # Load MCD samples if they exist

            if test_ensemble_logits is None:
                trained_model_paths = train_deep_ensemble(args, model_class, config, tokenizer, num_models=5)
                models = load_ensemble_models(model_class, config, tokenizer, trained_model_paths, args)
                # Get Deep Ensemble logit samples
                test_ensemble_logits = deep_ensemble_predictions(models, test_dataloader)  # Shape: (num_models, batch_size, num_classes)
                save_samples(method, test_ensemble_logits, ind = "test")

            test_sws = compute_sws(test_ensemble_logits)
            test_pv = compute_pv(test_ensemble_logits)
            test_bald = compute_bald(test_ensemble_logits)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph(test_sws.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_sws.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_before_scale" )

            scores_path = os.path.join(method_dir, "sws_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_sws, eval_correctness, test_sws, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, eval_sws, eval_correctness, test_sws, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_after_scale" )

            scores_path = os.path.join(method_dir, "sws_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            ####platt scaling
            plat_scaled = platt_scaling_pipeline(eval_sws, eval_correctness, test_sws, "sws",method_dir)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_sws, 1-eval_correctness, test_sws, "newuncertain_sws",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "sws_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "sws_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #pv

            ECE, ACC, MAX_ECE, brier_score = draw_reliability_graph((1-test_pv).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_pv.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_before_scale" )

            scores_path = os.path.join(method_dir, "pv_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_pv, eval_correctness, test_pv, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_pv, eval_correctness, 1-test_pv, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_after_scale" )

            scores_path = os.path.join(method_dir, "pv_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_pv, eval_correctness, 1-test_pv, "pv",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_pv, 1-eval_correctness, test_pv, "newuncertain_pv",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "pv_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "pv_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            #bald

            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((1-test_bald).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_bald.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_before_scale" )

            scores_path = os.path.join(method_dir, "bald_before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_bald, eval_correctness, test_bald, test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, 1-eval_bald, eval_correctness, 1-test_bald, test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_after_scale" )

            scores_path = os.path.join(method_dir, "bald_after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            """test_entropy1 = test_bald.cpu().numpy()

            with open(os.path.join("./saved_models/","before_bald_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,test_entropy1):
                    f.write(example.idx+'\t'+str(sc)+'\n')

            test_entropy2 = uce_scaled_values.cpu().numpy()

            with open(os.path.join("./saved_models/","after_bald_uncertainty.txt"),'w') as f:
                for example, sc in zip(test_dataset.examples,test_entropy2):
                    f.write(example.idx+'\t'+str(sc)+'\n')"""


            ####platt scaling
            plat_scaled = platt_scaling_pipeline(1-eval_bald, eval_correctness, 1-test_bald, "bald",method_dir)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            uncertain_plat_scaled = platt_scaling_pipeline(eval_bald, 1-eval_correctness, test_bald, "newuncertain_bald",method_dir)
            UCE,MAX_UCE = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "bald_plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "bald_platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")



        elif method == "dsmg":
            dissector = ImportanceScore(model=model, train_loader=eval_dataloader, dev_loader=test_dataloader, args=args)
            eval_scores = dissector._uncertainty_calculate(eval_dataloader)
            eval_scores = torch.tensor(eval_scores)
            eval_min = eval_scores.min()
            eval_max = eval_scores.max()
            eval_score_normalized = (eval_scores - eval_min) / (eval_max - eval_min + 1e-8)


            test_scores = dissector._uncertainty_calculate(test_dataloader)
            test_scores = torch.tensor(test_scores)

            test_min = test_scores.min()
            test_max = test_scores.max()
            test_score_normalized = (test_scores - eval_min) / (test_max - test_min + 1e-8)


            method_dir = os.path.join(ROOT_DIR, method)
            os.makedirs(method_dir, exist_ok=True)
            ECE, ACC, MAX_ECE,brier_score = draw_reliability_graph(test_score_normalized.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(test_score_normalized.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "before_scale" )

            scores_path = os.path.join(method_dir, "before_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")

            #apply temperature scaling

            uce_scaled_values = train_UCE_temperature(args, model, eval_score_normalized.to(device), eval_correctness, test_score_normalized.to(device), test_correctness)
            uce_scaled_values = torch.tensor(uce_scaled_values)
            ece_scaled_values = train_ECE_temperature(args, model, eval_score_normalized.to(device), eval_correctness, test_score_normalized.to(device), test_correctness)
            ece_scaled_values = torch.tensor(ece_scaled_values)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((ece_scaled_values).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )
            UCE,MAX_UCE  = draw_uce_reliability_graph(uce_scaled_values.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "after_scale" )

            scores_path = os.path.join(method_dir, "after_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n")
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")


            plat_scaled = platt_scaling_pipeline(eval_score_normalized, eval_correctness, test_score_normalized, "dsmg",method_dir)
            ECE, ACC,MAX_ECE, brier_score = draw_reliability_graph((plat_scaled).to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )


            uncertain_plat_scaled = platt_scaling_pipeline(eval_score_normalized, 1-eval_correctness, test_score_normalized, "newuncertain_dsmg",method_dir)
            UCE,MAX_UCE  = draw_uce_reliability_graph(uncertain_plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )
            #UCE  = draw_uce_reliability_graph(plat_scaled.to(device),test_label.to(device), test_preds.to(device), 10, method_dir, "plat_scaled" )

            scores_path = os.path.join(method_dir, "platscaled_scores.txt")
            with open(scores_path, "w") as f:
                f.write(f" ECE: {ECE:.4f}\n")
                f.write(f" UCE: {UCE:.4f}\n")
                f.write(f" Brier Score: {brier_score:.4f}\n")
                f.write(f" ACC: {ACC:.4f}\n") 
                f.write(f"MAX_ECE:{MAX_ECE:.4f}\n ")
                f.write(f"MAX_UCE:{MAX_UCE:.4f}\n ")










            








if __name__== '__main__':
    main()