from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler,RandomSampler
from torch.utils.data.distributed import DistributedSampler
import json


from tqdm import tqdm, trange
import multiprocessing
from model import Model, DecoderClassifier

cpu_cont = multiprocessing.cpu_count()

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
cpu_cont = multiprocessing.cpu_count()
from torch.optim import AdamW
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

logger = logging.getLogger(__name__)

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, get_scheduler,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer, 
                          AutoConfig, AutoModel, AutoTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'codellama' : (AutoConfig,AutoModel, AutoTokenizer),
    'deepseek' : (AutoConfig,AutoModel, AutoTokenizer),
    'qwen7b' : (AutoConfig,AutoModel, AutoTokenizer),
    'starcoder3b' : (AutoConfig,AutoModel, AutoTokenizer),
    'incoder1b' : (AutoConfig,AutoModel, AutoTokenizer),
    'codegemma' : (AutoConfig,AutoModel, AutoTokenizer)
    }


from plots import draw_uce_reliability_graph, draw_reliability_graph
from sklearn.metrics import roc_auc_score
#from emsemble import *
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
    code = js['input']
    if args.model_type in ["codellama"]:
        code_tokens = tokenizer.tokenize(code)
        if '</s>' in code_tokens:
            code_tokens = code_tokens[:code_tokens.index('</s>')]
        source_tokens = code_tokens[:args.block_size]
    elif args.model_type in ["starcoder3b", "deepseek", "incoder1b",'qwen7b']:
        code_tokens=tokenizer.tokenize(code)
        source_tokens = code_tokens[:args.block_size]
    else:
        code_tokens=tokenizer.tokenize(code)
        code_tokens = code_tokens[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    if args.model_type in ["codellama"]:
        source_ids = tokenizer.encode(js['input'].split("</s>")[0], max_length=args.block_size, padding='max_length', truncation=True)
    else:
        source_ids = tokenizer.encode(
            code,
            max_length=args.block_size,
            padding='max_length',
            truncation=True,
            add_special_tokens=False,  # decoder-only style
        )

        #source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        #padding_length = args.block_size - len(source_ids)
        #source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['id'],js['label'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

import random
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
                logits = model(input_ids=inputs)  # Get raw logits
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
    mc_probs = logits #F.softmax(logits, dim=-1)  # Convert logits to probabilities
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
    mc_probs = logits #F.softmax(logits, dim=-1)  # Convert logits to probabilities
    pv_scores = mc_probs.var(dim=0).mean(dim=1)  # Variance over MC samples, mean over classes


    pv_min = pv_scores.min()
    pv_max = pv_scores.max()
    pv_score_normalized = (pv_scores - pv_min) / (pv_max - pv_min + 1e-8)  # Small epsilon to prevent division by zero

    print(pv_score_normalized)
    return pv_score_normalized  # Higher variance = more uncertainty

def compute_bald(logits: torch.Tensor):
    """
    Compute BALD (Bayesian Active Learning by Disagreement) Score using PyTorch.

    Args:
        logits: Tensor of shape (iter_time, batch_size, num_classes), MC Dropout logits.

    Returns:
        bald_scores: BALD uncertainty scores (batch_size,).
    """
    mc_probs = logits #F.softmax(logits, dim=-1)  # Convert logits to probabilities

    # Compute entropy of the mean probability distribution
    mean_probs = mc_probs.mean(dim=0)  # Mean probability over MC samples (batch_size, num_classes)
    mean_entropy = -(mean_probs * mean_probs.log()).sum(dim=-1)  # Entropy of mean probabilities

    # Compute mean of entropies across MC samples
    sample_entropies = -(mc_probs * mc_probs.log()).sum(dim=-1)  # Entropy for each MC sample (iter_time, batch_size)
    mean_sample_entropy = sample_entropies.mean(dim=0)  # Mean entropy across MC samples (batch_size,)

    bald_score = mean_entropy -  mean_sample_entropy # BALD score # Higher = more epistemic uncertainty
    #bald_score_normalized = bald_score / torch.log(torch.tensor(2, dtype=torch.float32))
    bald_min = bald_score.min()
    bald_max = bald_score.max()
    bald_score_normalized = (bald_score - bald_min) / (bald_max - bald_min + 1e-8)  # Small epsilon to prevent division by zero

    return bald_score_normalized

    #return bald_score_clamped


def train_deep_ensemble(args, model_class, config, tokenizer, accelerator,num_models=5):
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
        seed = args.seed + i
        set_seed(seed)

        model_dir = os.path.join(base_dir, f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)
        args.output_dir = model_dir

        accelerator.print(f"\n>>> Training Ensemble Model {i+1}/{num_models} <<<\n")

        print(f"\nTraining Model {i+1}/{num_models} with seed {seed}...\n")

        compute_dtype = getattr(torch, "bfloat16")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
        model = model_class.from_pretrained(args.model_name_or_path,
                                                quantization_config=quant_config, trust_remote_code=True,
                                                torch_dtype = torch.bfloat16)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

             #config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        config.pad_token_id = tokenizer.pad_token_id
        model = DecoderClassifier(model,config,tokenizer,args)

        if args.model_type in ['starcoder3b', 'incoder1b']:
            peft_params = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type=TaskType.SEQ_CLS,   # sequence classification
                target_modules=["c_attn", "c_proj", "c_fc"]  # StarCoder modules
            )
        elif args.model_type in ['incoder1b']:
            peft_params = LoraConfig(
                task_type=TaskType.SEQ_CLS,   # decoder-only causal LM
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                # GPT-2/Incoder module names:
                target_modules=["c_attn", "c_proj", "c_fc"],
                fan_in_fan_out=True,                # important for GPT-2/Conv1D
        # keep classifier head (if any) trainable:
        
            )
        else:
            peft_params = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )

        model = get_peft_model(model, peft_params)
        #model.to(args.device)

        # Train model only when training is needed
        """with accelerator.main_process_first():
            train_dataset = TextDataset(tokenizer, args, args.train_data_file)"""
        #train_model(model, train_dataset,tokenizer,args, accelerator)

        """if accelerator.is_main_process:
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)"""

        trained_model_paths.append(model_dir)

        """# ✅ cleanup to really free GPU memory before next loop
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        accelerator.free_memory()"""

    return trained_model_paths


    """
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_dataloader):
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            loss,outputs = model(input_ids=inputs, labels=labels)
            #loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed.")

    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    """


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
        compute_dtype = getattr(torch, "bfloat16")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
        model = model_class.from_pretrained(args.model_name_or_path,
                                                quantization_config=quant_config, trust_remote_code=True,
                                                torch_dtype = torch.bfloat16)

        model = DecoderClassifier(model, config, tokenizer, args)  # Reinitialize model

        if args.model_type in ['starcoder3b', 'incoder1b']:
            peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type=TaskType.SEQ_CLS,   # sequence classification
            target_modules=["c_attn", "c_proj", "c_fc"]  # StarCoder modules
        )
        elif args.model_type in ['incoder1b']:
            peft_params = LoraConfig(
            task_type=TaskType.SEQ_CLS,   # decoder-only causal LM
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        # GPT-2/Incoder module names:
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True,                # important for GPT-2/Conv1D
        # keep classifier head (if any) trainable:
        
    )
        else:
            peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        model = get_peft_model(model, peft_params)

        #model.to("cuda")

        # Define model file path
        model_path = os.path.join(path, "model.pth")

        # Ensure the model file exists before loading
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path),strict=False)
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
                logits = model(input_ids=inputs)  # Get raw logits
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
                logits = model(input_ids=inputs)

                probs = logits #F.softmax(logits, dim=-1)
               
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
    ##agreement = (mc_preds == test_preds.unsqueeze(0)).float()  # Shape: [num_inferences, num_samples]
    ##agreement_ratio = agreement.mean(dim=0)
    ##mc_prob_variance = max_probs.var(dim=0)  # Shape: [num_samples]

   # Get top-2 probabilities for each MC inference
    top2_probs, _ = mc_probs.topk(2, dim=-1)  # Shape: [num_inferences, num_samples, 2]
    bvsb_correct = top2_probs[:, :, 0] - top2_probs[:, :, 1]  # BVSB for correct cases

# Get probability assigned to original model's prediction in MC Dropout
    #original_class_probs = mc_probs.gather(-1, test_probs.unsqueeze(0).unsqueeze(-1)).squeeze(-1)  # Shape: [num_inferences, num_samples]
    bvsb_incorrect = top2_probs[:, :, 0] - test_probs  # BVSB for incorrect cases

# Determine when MC aligns with original model prediction
    agreement = (mc_preds == test_preds.unsqueeze(0)).float()  # Shape: [num_inferences, num_samples]

# Compute validity score per inference
    validity_scores = agreement * (max_probs ) + (1 - agreement) * (max_probs-bvsb_correct)  # Shape: [num_inferences, num_samples]

    # Compute confidence-weighted alignment score
    #scores = agreement_ratio * (test_probs + mc_prob_variance)
   
    final_validity_scores = validity_scores.mean(dim=0)
# Convert to expected output shape torch.tensor([num_samples])
    final_scores = final_validity_scores.view(-1)  # Shape: [num_samples]


    """mc_prob_variance = mc_probs.var(dim=0).max(dim=-1)[0]  # Max variance per sample
    stability_factor = torch.exp(-1.5 * mc_prob_variance)  # Lower variance => Higher stability

# Confidence scaling factor (boost stable high-confidence cases)
    confidence_spread = (test_probs - 0.5) ** 2  # Spread confidence score
    scaling_factor = 1 + confidence_spread  # Confidence-aware adjustment

# Compute validity score per inference
    validity_scores = agreement * bvsb_correct + (1 - agreement) * bvsb_incorrect  # Shape: [num_inferences, num_samples]

    # Apply confidence scaling and stability weighting
    weighted_validity = validity_scores * scaling_factor * stability_factor

# Aggregate using a robust approach (median aggregation)
    final_validity_scores = weighted_validity.median(dim=0).values  # Robust against outliers

# Ensure output shape is torch.tensor([num_samples])
    final_scores = final_validity_scores.view(-1)  # Shape: [num_samples]"""



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


from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def get_threshold111(scores,labels, predictions):
    device = scores.device  # Ensure all tensors use the same device
    labels =labels.to(device)
    predictions =predictions.to(device)

    num_bins = 10
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=scores.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

# Initialize bin counts
    bin_correct = torch.zeros(num_bins, device=scores.device)
    bin_incorrect = torch.zeros(num_bins, device=scores.device)

# Calculate correctness of predictions
    correctness = predictions.eq(labels).float()

# Count correct and incorrect predictions in each bin
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (scores >= bin_lower) & (scores < bin_upper)  # Filter samples within the bin range
        if in_bin.any():  # Check if there are samples in the bin
            bin_correct[i] = correctness[in_bin].sum()  # Sum of correct predictions in bin
            bin_incorrect[i] = in_bin.sum() - bin_correct[i]  # Incorrect predictions in bin

# Convert results to a DataFrame
    df_bin_results = pd.DataFrame({
    "Score Bin": [f"{bin_lowers[i].item():.2f}-{bin_uppers[i].item():.2f}" for i in range(num_bins)],
    "Correct Predictions": bin_correct.cpu().numpy(),
    "Incorrect Predictions": bin_incorrect.cpu().numpy()
})
    print("success")

    return df_bin_results

def get_thresholdold(
    uncertainty,   # array-like, shape (N,) – higher = more uncertain
    labels,        # array-like, shape (N,) – ground-truth class ids
    preds,         # array-like, shape (N,) – predicted class ids
    min_frac=0.30, # minimum fraction of samples the selected group must cover
    n_thresholds=200,
):
    """
    For classification: compute correctness from labels & preds,
    then sweep thresholds on 'uncertainty' and pick the group (>= or <)
    with the highest misprediction rate, subject to a min coverage.

    Returns a dict with the chosen threshold and summary stats.
    """
    u = uncertainty.detach().cpu().numpy()
    u = np.asarray(u, dtype=float)
    y = labels.detach().cpu().numpy()
    y = np.asarray(y)
    yhat = preds.detach().cpu().numpy()
    yhat = np.asarray(yhat)

    if not (u.shape == y.shape == yhat.shape):
        raise ValueError("uncertainty, labels, preds must have the same shape.")

    # correctness (1 = correct, 0 = incorrect) and mispred (1 = incorrect)
    correct = (yhat == y).astype(int)
    mispred = 1 - correct

    # drop NaNs in uncertainty if any
    keep = ~np.isnan(u)
    u = u[keep]; mispred = mispred[keep]; correct = correct[keep]
    N = len(u)
    if N == 0:
        raise ValueError("No valid samples after filtering.")

    min_samples = max(1, int(np.ceil(min_frac * N)))
    lo, hi = float(np.min(u)), float(np.max(u))
    thresholds = np.array([lo]) if np.isclose(lo, hi) else np.linspace(lo, hi, n_thresholds)

    best = {
        "best_threshold": None,
        "group": None,  # "above" (u >= t) or "below" (u < t)
        "misp_rate_pct": -1.0,
        "misp_count": 0,
        "total_in_group": 0,
        "coverage_pct": 0.0,
        "global_misp_rate_pct": 100.0 * mispred.mean(),
        "n_samples": N,
    }

    for t in thresholds:
        idx_hi = (u >= t)
        idx_lo = ~idx_hi

        for name, idx in (("above", idx_hi), ("below", idx_lo)):
            k = int(idx.sum())
            if k < min_samples:
                continue
            rate = float(mispred[idx].mean())  # fraction wrong in this group
            # prefer higher mispred rate; break ties by larger coverage
            better = rate > (best["misp_rate_pct"] / 100.0)
            tie = np.isclose(rate, best["misp_rate_pct"] / 100.0) and k > best["total_in_group"]
            if better or tie:
                best.update({
                    "best_threshold": float(t),
                    "group": name,
                    "misp_rate_pct": 100.0 * rate,
                    "misp_count": int(mispred[idx].sum()),
                    "total_in_group": k,
                    "coverage_pct": 100.0 * k / N,
                })

    # If nothing met coverage, fall back to reporting global stats
    if best["best_threshold"] is None:
        best.update({
            "best_threshold": float(hi),
            "group": "all",
            "misp_rate_pct": 100.0 * mispred.mean(),
            "misp_count": int(mispred.sum()),
            "total_in_group": N,
            "coverage_pct": 100.0,
        })

    # Also return masks at the chosen threshold for downstream use
    t = best["best_threshold"]
    idx_above = (u >= t)
    idx_below = ~idx_above
    best["mask_above"] = idx_above
    best["mask_below"] = idx_below
    best["correct"] = correct  # 1 = correct, 0 = incorrect
    best["mispred"] = mispred  # 1 = incorrect, 0 = correct
    return best



def get_threshold(
    uncertainty,   # array-like, shape (N,) – higher = more uncertain
    labels,        # array-like, shape (N,) – ground-truth class ids
    preds,         # array-like, shape (N,) – predicted class ids
    min_frac=0.30, # minimum fraction of samples the selected group must cover
    n_thresholds=200,
    mode="misp",   # "misp" = group with highest mispred rate, "corr" = highest correct rate
):
    """
    Sweep thresholds on 'uncertainty' and pick the group (>= or < threshold)
    that maximizes either misprediction rate ("misp") or correctness rate ("corr"),
    subject to a minimum coverage constraint.

    Returns summary stats: counts of correct/incorrect in the group,
    and their fractions relative to global totals.
    """
    u = np.asarray(uncertainty.detach().cpu().numpy(), dtype=float)
    y = np.asarray(labels.detach().cpu().numpy())
    yhat = np.asarray(preds.detach().cpu().numpy())

    if not (u.shape == y.shape == yhat.shape):
        raise ValueError("uncertainty, labels, preds must have the same shape.")

    # correctness (1 = correct, 0 = incorrect)
    correct = (yhat == y).astype(int)
    mispred = 1 - correct

    # drop NaNs in uncertainty if any
    keep = ~np.isnan(u)
    u, correct, mispred = u[keep], correct[keep], mispred[keep]
    N = len(u)
    if N == 0:
        raise ValueError("No valid samples after filtering.")

    min_samples = max(1, int(np.ceil(min_frac * N)))
    lo, hi = float(np.min(u)), float(np.max(u))
    thresholds = np.array([lo]) if np.isclose(lo, hi) else np.linspace(lo, hi, n_thresholds)

    total_correct = int(correct.sum())
    total_misp = int(mispred.sum())

    best = {
        "best_threshold": None,
        "group": None,
        "n_correct_in_group": 0,
        "n_misp_in_group": 0,
        "frac_correct_of_total": 0.0,
        "frac_misp_of_total": 0.0,
        "total_in_group": 0,
        "coverage_pct": 0.0,
        "global_correct": total_correct,
        "global_misp": total_misp,
        "n_samples": N,
    }

    for t in thresholds:
        idx_hi = (u >= t)
        idx_lo = ~idx_hi

        for name, idx in (("above", idx_hi), ("below", idx_lo)):
            k = int(idx.sum())
            if k < min_samples:
                continue

            n_corr = int(correct[idx].sum())
            n_misp = int(mispred[idx].sum())
            rate_corr = n_corr / k if k > 0 else 0.0
            rate_misp = n_misp / k if k > 0 else 0.0

            if mode == "misp":
                score = rate_misp
                best_score = best["n_misp_in_group"] / best["total_in_group"] if best["total_in_group"] > 0 else -1
            elif mode == "corr":
                score = rate_corr
                best_score = best["n_correct_in_group"] / best["total_in_group"] if best["total_in_group"] > 0 else -1
            else:
                raise ValueError("mode must be 'misp' or 'corr'")

            better = score > best_score
            tie = np.isclose(score, best_score) and k > best["total_in_group"]

            if better or tie:
                best.update({
                    "best_threshold": float(t),
                    "group": name,
                    "n_correct_in_group": n_corr,
                    "n_misp_in_group": n_misp,
                    "frac_correct_of_total": 100.0 * n_corr / max(1, total_correct),
                    "frac_misp_of_total": 100.0 * n_misp / max(1, total_misp),
                    "total_in_group": k,
                    "coverage_pct": 100.0 * k / N,
                })

    # Fallback: if nothing met coverage, just return global stats
    if best["best_threshold"] is None:
        best.update({
            "best_threshold": float(hi),
            "group": "all",
            "n_correct_in_group": total_correct,
            "n_misp_in_group": total_misp,
            "frac_correct_of_total": 100.0,
            "frac_misp_of_total": 100.0,
            "total_in_group": N,
            "coverage_pct": 100.0,
        })

    # Also return masks for downstream use
    t = best["best_threshold"]
    idx_above = (u >= t)
    idx_below = ~idx_above
    best["mask_above"] = idx_above
    best["mask_below"] = idx_below
    best["correct"] = correct
    best["mispred"] = mispred
    return best
    
    
    
    
    
    
    
    # Convert tensors to CPU for processing
    """scores_cpu = scores.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

# Define bins dynamically based on actual score range
    num_bins = 10
    min_score, max_score = scores_cpu.min(), scores_cpu.max()
    bin_edges = np.linspace(min_score, max_score, num_bins + 1)

# Assign each score to a bin
    binned_indices = np.digitize(scores_cpu, bin_edges) - 1

# Initialize a dictionary for storing counts per bin
    bin_results = {f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}": {"Correct": 0, "Incorrect": 0} 
               for i in range(len(bin_edges) - 1)}

    # Count correct and incorrect predictions in each bin
    for i, bin_idx in enumerate(binned_indices):
        if 0 <= bin_idx < len(bin_edges) - 1:
            bin_label = f"{bin_edges[bin_idx]:.2f}-{bin_edges[bin_idx + 1]:.2f}"
            if predictions_cpu[i] == labels_cpu[i]:
                bin_results[bin_label]["Correct"] += 1
            else:
                bin_results[bin_label]["Incorrect"] += 1

# Convert to DataFrame
    df_bin_results = pd.DataFrame.from_dict(bin_results, orient="index").reset_index()

    print(df_bin_results)
    

    # Define bins for thresholds (10 bins from 0.0 to 1.0)
    bin_edges = torch.linspace(0, 1, 11, device=device)  # 11 edges create 10 bins
    bin_labels = [(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}") for i in range(len(bin_edges)-1)]

    # Assign each score to a bin
    binned_indices = torch.bucketize(scores, bin_edges) - 1  # Get bin indices

    # Initialize counts dictionary
    bin_results = {label: {"Correct": 0, "Incorrect": 0} for label in bin_labels}

    # Count correct and incorrect predictions in each bin
    for i, bin_idx in enumerate(binned_indices.cpu().numpy()):
        bin_label = bin_labels[bin_idx] if 0 <= bin_idx < len(bin_labels) else None
        if bin_label:
            if predictions[i] == labels[i]:
                bin_results[bin_label]["Correct"] += 1
            else:
                bin_results[bin_label]["Incorrect"] += 1

    # Convert results to DataFrame
    df_bin_results = pd.DataFrame.from_dict(bin_results, orient="index")
    df_bin_results.reset_index(inplace=True)
    df_bin_results.rename(columns={"index": "Score Bin"}, inplace=True)
    print(df_bin_results)"""

    """device = scores.device  # Ensure all tensors are on the same device

    # Compute correctness (1 if prediction is correct, 0 otherwise)
    correct = (predictions == labels).int().cpu().numpy()
    scores_np = scores.cpu().numpy()  # Convert to NumPy for sklearn

    # Compute ROC curve to get FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(correct, scores_np)

    # Compute Youden's J statistic (J = TPR - FPR)
    youden_j = tpr - fpr
    best_idx = youden_j.argmax()
    best_threshold = thresholds[best_idx]

    # Count correct and incorrect predictions beyond the threshold
    final_preds_above_threshold = scores <= best_threshold
    correct_final = correct[final_preds_above_threshold.cpu()].sum()
    incorrect_final = final_preds_above_threshold.sum().item() - correct_final

    print(best_threshold, correct_final, incorrect_final)

    return best_threshold, correct_final, incorrect_final"""

    
metrics_rows = []
bins_rows = []

def _to_float(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return x

def record_result(method, stage, ECE, ACC, MAX_ECE, brier_score, UCE=None, MAX_UCE=None, uncertain_bin=None):
    metrics_rows.append({
        "method": method, "stage": stage,
        "ECE": _to_float(ECE), "MAX_ECE": _to_float(MAX_ECE),
        "UCE": _to_float(UCE) if UCE is not None else None,
        "MAX_UCE": _to_float(MAX_UCE) if MAX_UCE is not None else None,
        "Brier": _to_float(brier_score), "ACC": _to_float(ACC),
    })
    if isinstance(uncertain_bin, dict):
        for k, v in uncertain_bin.items():
            bins_rows.append({"method": method, "stage": stage, "bin": str(k), "value": _to_float(v)})

def write_excel_per_method(method, method_dir, one_sheet=True):
    """Write an Excel file for this method into its directory."""
    mdf = pd.DataFrame([r for r in metrics_rows if r["method"] == method])
    bdf = pd.DataFrame([r for r in bins_rows    if r["method"] == method])

    os.makedirs(method_dir, exist_ok=True)
    xlsx_path = os.path.join(method_dir, f"{method}_calibration.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        if one_sheet:
            # one sheet named 'summary': metrics first, then bins below
            mdf.to_excel(w, index=False, sheet_name="summary", startrow=0)
            startrow = len(mdf) + 2
            # small label row
            pd.DataFrame({"section": ["uncertainty_bins"]}).to_excel(
                w, index=False, sheet_name="summary", startrow=startrow, header=False
            )
            bdf.to_excel(w, index=False, sheet_name="summary", startrow=startrow + 1)
        else:
            # two sheets
            mdf.to_excel(w, index=False, sheet_name="metrics")
            bdf.to_excel(w, index=False, sheet_name="uncertainty_bins")

    print(f"[OK] Wrote {xlsx_path}")





import os, csv, json, numpy as np, torch
import torch.nn as nn

# =========================
# Converters
# =========================
def _to_numpy(x):
    if isinstance(x, np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _np1(x): a = _to_numpy(x); return a.reshape(-1)

def _np2(x):
    a = _to_numpy(x)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got {a.shape}")
    return a

# =========================
# Scoring & helpers
# =========================
def sr_score_from_probs(probs):     # MSP = max softmax prob
    P = _np2(probs)
    return P.max(axis=1)

def preds_from_probs(probs):
    P = _np2(probs)
    return P.argmax(axis=1)

def _accept_masks_exact(scores, coverages):
    """Exact top-k accept masks for each coverage (no interpolation)."""
    s = _np1(scores); N = len(s)
    order = np.argsort(-s)
    s_sorted = s[order]
    masks, taus = {}, {}
    for c in map(float, coverages):
        k = int(np.floor(c * N)); k = min(max(k, 1), N)
        keep = np.zeros(N, dtype=bool); keep[order[:k]] = True
        masks[c] = keep
        taus[c] = float(s_sorted[k-1])
    return masks, taus

# =========================
# Metrics
# =========================
def _overall_ece(probs_task, y_true, yhat, n_bins=15):
    """ECE over ALL samples (not only accepted)."""
    P  = _np2(probs_task)
    y  = _np1(y_true)
    yh = _np1(yhat)

    if len(y) == 0: return float("nan")
    conf = P[np.arange(len(y)), yh]
    corr = (yh == y).astype(np.float32)

    bins = np.linspace(0., 1., n_bins+1)
    ece  = 0.0; total = len(conf)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1] if i < n_bins-1 else conf <= bins[i+1])
        if m.any():
            ece += (m.sum()/total) * abs(corr[m].mean() - conf[m].mean())
    return float(ece)

def rc_curve_and_aurc_by_scores(y_true, scores, yhat):
    """
    Full prefix RC curve (k=1..N) + AURC, oracle AURC.
    """
    y  = _np1(y_true)
    s  = _np1(scores)
    yh = _np1(yhat)
    N  = len(y)
    if N == 0:
        return {"coverage_curve": np.array([]), "risk_curve": np.array([]), "acc_curve": np.array([]),
                "AURC": float("nan"), "AURC_oracle": float("nan"), "E_AURC": float("nan"), "order": np.array([])}

    correct = (yh == y).astype(np.int32)
    order   = np.argsort(-s)      # high→low
    corr_sorted = correct[order]

    k = np.arange(1, N+1)
    cov  = k / N
    acc  = np.cumsum(corr_sorted) / k
    risk = 1.0 - acc

    aurc = float(np.trapz(risk, cov))

    corr_or = np.sort(correct)[::-1]
    acc_or  = np.cumsum(corr_or) / k
    risk_or = 1.0 - acc_or
    aurc_or = float(np.trapz(risk_or, cov))

    return {
        "coverage_curve": cov,        # dense (1/N, 2/N, ..., 1)
        "risk_curve": risk,
        "acc_curve": acc,
        "AURC": aurc,
        "AURC_oracle": aurc_or,
        "E_AURC": aurc - aurc_or,
        "order": order
    }

def _rc_on_exact_grid_from_prefix(prefix_cov, prefix_acc, coverage_grid):
    """
    Linear interpolation of selective accuracy at requested coverage points.
    coverage_grid must be within (0,1]; e.g., [0.10, 0.20, ..., 1.00].
    """
    cov = np.asarray(prefix_cov, dtype=float)
    acc = np.asarray(prefix_acc, dtype=float)
    grid = np.asarray(coverage_grid, dtype=float)
    grid = np.clip(grid, cov[0], cov[-1])
    acc_grid = np.interp(grid, cov, acc)
    risk_grid = 1.0 - acc_grid
    return grid, acc_grid, risk_grid

# =========================
# CSV Writers
# =========================
def _write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)
    return path

def save_rc_curve_csv(save_dir, base_name, rc):
    """Assumes rc contains the EXACT grid arrays."""
    cov = _np1(rc["coverage_curve"]); risk = _np1(rc["risk_curve"]); acc = _np1(rc["acc_curve"])
    rows = list(zip(cov, risk, acc))
    header = ["coverage","risk","accuracy"]
    return _write_csv(os.path.join(save_dir, f"{base_name}_rc_curve.csv"), rows, header)

def save_meta_csv(save_dir, base_name, meta):
    """One-row CSV with plain acc, overall ECE, AURC, AURC_oracle, E_AURC."""
    rows = [[
        float(meta.get("plain_task_acc", np.nan)),
        float(meta.get("overall_ECE", np.nan)),
        float(meta.get("AURC", np.nan)),
        float(meta.get("AURC_oracle", np.nan)),
        float(meta.get("E_AURC", np.nan)),
    ]]
    header = ["plain_task_acc","overall_ECE","AURC","AURC_oracle","E_AURC"]
    return _write_csv(os.path.join(save_dir, f"{base_name}_meta.csv"), rows, header)

def save_per_class_true_csv(save_dir, base_name, rows):
    header = ["class","coverage","sel_acc_cls","sel_risk_cls","n_accept_cls","n_total_cls"]
    return _write_csv(os.path.join(save_dir, f"{base_name}_per_class_acc_risk.csv"), rows, header)

def save_predictions_files(save_dir, base_name, ids_test, yhat, scores, accept_masks_by_c=None):
    """Optional: saves predictions and detailed accept flags if ids_test provided."""
    if ids_test is None: 
        return
    ids = list(ids_test)
    yh  = _np1(yhat)
    sc  = _np1(scores)
    _write_csv(os.path.join(save_dir, f"{base_name}_predictions.txt"),
               [[ids[i], int(yh[i])] for i in range(len(ids))],
               header=None)
    if accept_masks_by_c is not None:
        covs = sorted(map(float, accept_masks_by_c.keys()))
        rows = []
        keeps = {c: _np1(accept_masks_by_c[c]).astype(int) for c in covs}
        header = ["idx","pred_task","selector_score"] + [f"accept@{c:.2f}" for c in covs]
        for i in range(len(ids)):
            row = [ids[i], int(yh[i]), float(sc[i])] + [int(keeps[c][i]) for c in covs]
            rows.append(row)
        _write_csv(os.path.join(save_dir, f"{base_name}_predictions_detailed.csv"), rows, header)

# =========================
# Per-class (TRUE) rows builder
# =========================
def _per_class_rows_true(y_true, yhat, accept_masks_by_c, n_classes):
    y  = _np1(y_true)
    yh = _np1(yhat)
    rows = []
    for c in sorted(accept_masks_by_c.keys()):
        keep = _np1(accept_masks_by_c[c]).astype(bool)
        for cls in range(n_classes):
            total_mask = (y == cls)
            kept_mask  = total_mask & keep
            n_total    = int(total_mask.sum())
            n_accept   = int(kept_mask.sum())
            if n_accept > 0:
                sel_acc  = float((yh[kept_mask] == y[kept_mask]).mean())
                sel_risk = 1.0 - sel_acc
            else:
                sel_acc = float("nan"); sel_risk = float("nan")
            rows.append([int(cls), float(c), sel_acc, sel_risk, n_accept, n_total])
    return rows

# =========================
# Baseline 1: SR / MSP (TEST-CAL exact-grid, CSV output)
# =========================
def run_sr_baseline_testcal_csv(test_probs, y_test, ids_test,
                                coverages=None, save_dir="outputs_sr_testcal"):
    """
    SR/MSP baseline. RC via interpolation from prefix; plus per-class via exact top-k masks.
    Saves: *_meta.csv, *_rc_curve.csv, *_per_class_acc_risk.csv, predictions (if ids provided).
    """
    test_probs = _np2(test_probs)
    y_test     = _np1(y_test)
    if coverages is None:
        coverages = [round(x, 2) for x in np.arange(0.10, 1.001, 0.10)]

    scores_test = sr_score_from_probs(test_probs)   # MSP
    yhat_test   = preds_from_probs(test_probs)
    n_classes   = test_probs.shape[1]

    # Dense prefix RC and AURC
    rc_dense = rc_curve_and_aurc_by_scores(y_test, scores_test, yhat_test)

    # Resample to EXACT grid for curve CSV
    cov_g, acc_g, risk_g = _rc_on_exact_grid_from_prefix(
        rc_dense["coverage_curve"], rc_dense["acc_curve"], coverages
    )
    rc_exact = {
        "coverage_curve": cov_g,
        "risk_curve": risk_g,
        "acc_curve": acc_g,
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    meta = {
        "plain_task_acc": float((yhat_test == y_test).mean()) if len(y_test) else float("nan"),
        "overall_ECE": _overall_ece(test_probs, y_test, yhat_test, n_bins=15),
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    base = "sr_testcal"
    os.makedirs(save_dir, exist_ok=True)
    save_meta_csv(save_dir, base, meta)
    save_rc_curve_csv(save_dir, base, rc_exact)

    # Per-class via exact top-k
    accept_masks_by_c, taus = _accept_masks_exact(scores_test, coverages)
    rows = _per_class_rows_true(y_test, yhat_test, accept_masks_by_c, n_classes)
    save_per_class_true_csv(save_dir, base, rows)

    acc_risk_rows = _acc_risk_rows(
    y_true=y_test,
    yhat=yhat_test,
    probs_task=test_probs,        # SR uses original probs
    accept_masks_by_c=accept_masks_by_c,
    taus=taus
    )
    _save_acc_risk_table_csv(save_dir, base, acc_risk_rows)

    # Optional predictions files
    save_predictions_files(save_dir, base, ids_test, yhat_test, scores_test, accept_masks_by_c)

    # Also save taus (useful for debugging thresholds)
    with open(os.path.join(save_dir, f"{base}_taus.json"), "w") as f:
        json.dump({f"{c:.2f}": t for c, t in sorted(taus.items())}, f, indent=2)

    return {"rc": rc_exact, "meta": meta}

# =========================
# Baseline 2: MCD-PV (TEST-CAL exact-grid, CSV output)
# =========================
def run_mcd_pv_baseline_testcal_csv(test_pv, test_probs, y_test, ids_test,
                                    coverages=None, save_dir="outputs_mcd_pv_testcal"):
    """
    MC-Dropout PV baseline. Score = -PV (higher = more confident).
    Saves: *_meta.csv, *_rc_curve.csv, *_per_class_acc_risk.csv, predictions (if ids provided).
    """
    test_pv    = _np1(test_pv)
    test_probs = _np2(test_probs)
    y_test     = _np1(y_test)
    if coverages is None:
        coverages = [round(x, 2) for x in np.arange(0.10, 1.001, 0.10)]

    scores_test = -test_pv
    yhat_test   = preds_from_probs(test_probs)
    n_classes   = test_probs.shape[1]

    rc_dense = rc_curve_and_aurc_by_scores(y_test, scores_test, yhat_test)

    cov_g, acc_g, risk_g = _rc_on_exact_grid_from_prefix(
        rc_dense["coverage_curve"], rc_dense["acc_curve"], coverages
    )
    rc_exact = {
        "coverage_curve": cov_g,
        "risk_curve": risk_g,
        "acc_curve": acc_g,
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    meta = {
        "plain_task_acc": float((yhat_test == y_test).mean()) if len(y_test) else float("nan"),
        "overall_ECE": _overall_ece(test_probs, y_test, yhat_test, n_bins=15),
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    base = "mcd_pv_testcal"
    os.makedirs(save_dir, exist_ok=True)
    save_meta_csv(save_dir, base, meta)
    save_rc_curve_csv(save_dir, base, rc_exact)

    # Per-class via exact top-k
    accept_masks_by_c, taus = _accept_masks_exact(scores_test, coverages)
    rows = _per_class_rows_true(y_test, yhat_test, accept_masks_by_c, n_classes)
    save_per_class_true_csv(save_dir, base, rows)

    acc_risk_rows = _acc_risk_rows(
    y_true=y_test,
    yhat=yhat_test,
    probs_task=test_probs,        # MCD metrics computed on original probs
    accept_masks_by_c=accept_masks_by_c,
    taus=taus
)
    _save_acc_risk_table_csv(save_dir, base, acc_risk_rows)

    # Optional predictions files
    save_predictions_files(save_dir, base, ids_test, yhat_test, scores_test, accept_masks_by_c)

    with open(os.path.join(save_dir, f"{base}_taus.json"), "w") as f:
        json.dump({f"{c:.2f}": t for c, t in sorted(taus.items())}, f, indent=2)

    return {"rc": rc_exact, "meta": meta}

# =========================
# Platt on MSP (fit on EVAL, apply on TEST)
# =========================
def _clip_probs(p, eps=1e-6):
    return torch.clamp(p, eps, 1. - eps)

class PlattBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    def forward(self, x):
        return torch.sigmoid(self.a * x + self.b)

@torch.no_grad()
def _msp_and_pred(probs: torch.Tensor):
    return probs.max(dim=-1)  # (msp, yhat)

def fit_platt_on_msp(eval_probs: torch.Tensor,
                     eval_label: torch.Tensor,
                     steps: int = 2000,
                     lr: float = 0.05,
                     weight_decay: float = 0.0,
                     batch_size: int = 0,
                     use_logit_input: bool = False) -> nn.Module:
    """
    Robust MSP→P(correct) logistic fit with Adam (no LBFGS).
    - eval_probs: [N, C] probabilities (TASK classes only if you use abstain externally)
    - eval_label: [N] int labels
    """
    device = eval_probs.device
    eval_probs = _clip_probs(eval_probs)
    msp_eval, yhat_eval = _msp_and_pred(eval_probs)      # [N]
    y_eval = (yhat_eval == eval_label).float()           # [N]

    x = msp_eval
    if use_logit_input:
        x = torch.log(x) - torch.log(1.0 - x)
    x = x.view(-1, 1)                                    # [N,1]
    y = y_eval.view(-1, 1)                               # [N,1]

    model = PlattBinary().to(device)
    bce = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    N = x.size(0)
    if batch_size is None or batch_size <= 0 or batch_size > N:
        batch_size = N  # full-batch by default

    idx = torch.arange(N, device=device)

    for _ in range(steps):
        with torch.enable_grad():
            perm = idx if batch_size == N else idx[torch.randint(0, N, (batch_size,), device=device)]
            xb = x[perm]
            yb = y[perm]
            pred = model(xb)                 # [B,1]
            loss = bce(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model

@torch.no_grad()
def apply_platt_msp(model: nn.Module, probs: torch.Tensor):
    """
    Calibrate the predicted-class probability using MSP->P(correct),
    then redistribute the remaining mass to non-predicted classes proportionally.
    Works for binary and multiclass.
    """
    probs = _clip_probs(probs)
    msp, yhat = _msp_and_pred(probs)                # [N], [N]
    c_cal = model(msp.unsqueeze(-1)).squeeze(-1)    # [N] calibrated confidence for argmax class
    c_cal = _clip_probs(c_cal)

    N, C = probs.shape
    probs_cal = probs.clone()
    for i in range(N):
        k = int(yhat[i])
        p_max = float(probs[i, k])
        rem = max(1.0 - p_max, 1e-12)
        scale = float((1.0 - float(c_cal[i])) / rem)
        if C > 1:
            for j in range(C):
                if j != k:
                    probs_cal[i, j] = probs[i, j] * scale
        probs_cal[i, k] = c_cal[i]

    # safety renorm
    probs_cal = probs_cal / probs_cal.sum(dim=-1, keepdim=True)
    return probs_cal

def _slice_task_probs(P, exclude_last_class):
    """Optionally drop the last column (abstain) before MSP/argmax."""
    if torch.is_tensor(P):
        C = P.size(1)
        return P[:, :C-1] if (exclude_last_class and C >= 2) else P
    else:
        P = _np2(P)
        C = P.shape[1]
        return P[:, :C-1] if (exclude_last_class and C >= 2) else P

def run_platt_msp_baseline_testcal_csv(
    eval_probs, eval_labels,
    test_probs, test_labels,
    ids_test,
    coverages,
    save_dir,
    exclude_last_class=True,
    base_name="ps_msp_testcal"
):
    """
    MSP-Platt baseline (fit on EVAL, apply on TEST).
    Saves: *_meta.csv, *_rc_curve.csv, *_per_class_acc_risk.csv, predictions (if ids provided), *_taus.json
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Slice to TASK probs if needed (drop abstain)
    eval_P_task = _slice_task_probs(eval_probs, exclude_last_class)
    test_P_task = _slice_task_probs(test_probs, exclude_last_class)

    # Ensure torch tensors
    device = eval_P_task.device if torch.is_tensor(eval_P_task) else torch.device("cpu")
    if not torch.is_tensor(eval_P_task): eval_P_task = torch.tensor(eval_P_task, device=device, dtype=torch.float32)
    if not torch.is_tensor(test_P_task): test_P_task = torch.tensor(test_P_task, device=device, dtype=torch.float32)
    if not torch.is_tensor(eval_labels): eval_labels = torch.tensor(_np1(eval_labels), device=device, dtype=torch.long)
    if not torch.is_tensor(test_labels): test_labels = torch.tensor(_np1(test_labels), device=device, dtype=torch.long)

    # 2) Fit on EVAL MSP → apply to TEST
    ps_model = fit_platt_on_msp(eval_P_task, eval_labels)
    with torch.no_grad():
        test_P_task_ps = apply_platt_msp(ps_model, test_P_task)  # [N, C_task]
        scores_test    = test_P_task_ps.max(dim=1).values        # MSP after calibration
        yhat_test      = test_P_task_ps.argmax(dim=1)

    y_true_np  = _np1(test_labels)
    yhat_np    = _np1(yhat_test)
    scores_np  = _np1(scores_test)
    n_classes  = test_P_task_ps.shape[1]

    # 3) Dense RC + interpolation for curve CSV
    rc_dense = rc_curve_and_aurc_by_scores(y_true_np, scores_np, yhat_np)
    cov_g, acc_g, risk_g = _rc_on_exact_grid_from_prefix(
        rc_dense["coverage_curve"], rc_dense["acc_curve"], coverages
    )
    rc_exact = {
        "coverage_curve": cov_g,
        "risk_curve": risk_g,
        "acc_curve": acc_g,
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    meta = {
        "plain_task_acc": float((yhat_np == y_true_np).mean()) if len(y_true_np) else float("nan"),
        "overall_ECE": _overall_ece(_np2(test_P_task_ps), y_true_np, yhat_np, n_bins=15),
        "AURC": rc_dense["AURC"],
        "AURC_oracle": rc_dense["AURC_oracle"],
        "E_AURC": rc_dense["E_AURC"],
    }

    # 4) Save meta & RC curve
    save_meta_csv(save_dir, base_name, meta)
    save_rc_curve_csv(save_dir, base_name, rc_exact)

    # 5) Per-class via exact top-k masks
    accept_masks_by_c, taus = _accept_masks_exact(scores_np, coverages)
    rows = _per_class_rows_true(y_true_np, yhat_np, accept_masks_by_c, n_classes)
    save_per_class_true_csv(save_dir, base_name, rows)

    acc_risk_rows = _acc_risk_rows(
    y_true=y_true_np,
    yhat=yhat_np,
    probs_task=_np2(test_P_task_ps),   # calibrated probs (tensor -> np)
    accept_masks_by_c=accept_masks_by_c,
    taus=taus
)
    _save_acc_risk_table_csv(save_dir, base_name, acc_risk_rows)

    # 6) Optional predictions + detailed accept flags
    save_predictions_files(save_dir, base_name, ids_test, yhat_np, scores_np, accept_masks_by_c)

    # 7) Taus JSON (kth score per coverage)
    with open(os.path.join(save_dir, f"{base_name}_taus.json"), "w") as f:
        json.dump({f"{c:.2f}": t for c, t in sorted(taus.items())}, f, indent=2)

    return {"rc": rc_exact, "meta": meta}

def ece_on_accepted(probs_task, y_true, yhat, accept_mask, n_bins=15):
    P    = _np2(probs_task)
    y    = _np1(y_true)
    yh   = _np1(yhat)
    keep = _np1(accept_mask).astype(bool)
    if keep.sum() == 0: 
        return float("nan")
    conf = P[np.arange(len(y)), yh]
    conf = conf[keep]
    yk   = (yh[keep] == y[keep]).astype(np.float32)
    bins = np.linspace(0., 1., n_bins+1)
    ece  = 0.0; total = len(conf)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1] if i < n_bins-1 else conf <= bins[i+1])
        if m.any():
            ece += (m.sum()/total) * abs(yk[m].mean() - conf[m].mean())
    return float(ece)

def _acc_risk_rows(y_true, yhat, probs_task, accept_masks_by_c, taus):
    """Builds the detailed accuracy–risk table rows for each coverage."""
    y  = _np1(y_true)
    yh = _np1(yhat)
    P  = _np2(probs_task)
    N  = len(y)
    rows = []
    for c in sorted(accept_masks_by_c.keys()):
        keep = _np1(accept_masks_by_c[c]).astype(bool)
        k    = int(keep.sum())
        sel_acc  = float((yh[keep] == y[keep]).mean()) if k > 0 else float("nan")
        sel_risk = (1.0 - sel_acc) if k > 0 else float("nan")
        ece_keep = ece_on_accepted(P, y, yh, keep, n_bins=15)
        rows.append([
            f"{float(c):.2f}",
            float(taus[c]),
            sel_acc,
            sel_risk,
            int(k),
            int(N),
            ece_keep
        ])
    return rows

def _save_acc_risk_table_csv(save_dir, base_name, rows):
    header = ["coverage","tau","selective_accuracy","selective_risk","n_accept","n_total","ECE_on_accepted"]
    path = os.path.join(save_dir, f"{base_name}_acc_risk.csv")
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    return path



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
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

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) 
    device = accelerator.device
    args.n_gpu = accelerator.num_processes
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size 
    args.per_gpu_eval_batch_size=args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("device: %s, n_gpu: %s, distributed training: %s",
                   device, args.n_gpu, bool(args.n_gpu > 1))
    #logger.info(accelerator.state, main_process_only=False)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'model.bin')
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
    
    config.num_labels = 4
    if args.model_type not in ["codellama"]:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                    trust_remote_code=True)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    compute_dtype = getattr(torch, "bfloat16")
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

    if args.model_name_or_path:
        if args.model_type in ["starcoder3b", "deepseek","incoder1b"]:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                quantization_config=quant_config, trust_remote_code=True,
                                                torch_dtype = torch.bfloat16)
                                                #attn_implementation = "flash_attention_2")
        else:
            print("load from here")
            model = model_class.from_pretrained(args.model_name_or_path,
                                                torch_dtype = torch.bfloat16, quantization_config=quant_config, trust_remote_code=True)
    else:
        model = model_class(config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    config.pad_token_id = tokenizer.pad_token_id

    #config.pad_token_id = tokenizer(tokenizer.pad_token, truncation=True)['input_ids'][0]
    if args.model_type in ['codellama', 'starcoder3b', "deepseek","incoder1b",'qwen7b']:
        model = DecoderClassifier(model,config,tokenizer,args)
    else:
        model = Model(model,config,tokenizer,args)

    logger.info("Training/evaluation parameters %s", args)
    
    if args.model_type in ['starcoder3b', 'incoder1b']:
        peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=TaskType.SEQ_CLS,   # sequence classification
        target_modules=["c_attn", "c_proj", "c_fc"]  # StarCoder modules
    )
    elif args.model_type in ['incoder1b']:
        peft_params = LoraConfig(
        task_type=TaskType.SEQ_CLS,   # decoder-only causal LM
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        # GPT-2/Incoder module names:
        target_modules=["c_attn", "c_proj", "c_fc"],
        fan_in_fan_out=True,                # important for GPT-2/Conv1D
        # keep classifier head (if any) trainable:
        
)
    else:
        peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model, peft_params)
    


    logger.info("Training/evaluation parameters %s", args)




    checkpoint_prefix = f'checkpoint-best-acc/{args.project}/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    model.load_state_dict(torch.load(output_dir), strict=False)                  
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
    print(eval_correctness)
    eval_erroneousness = 1 - eval_correctness  # Complement of correctness
    test_probs, test_label, test_preds = get_predictions(model, test_dataloader, args)
    test_correctness = (test_preds == test_label).float()  # 1 for correct, 0 for incorrect
    test_erroneousness = 1 - test_correctness  # Complement of correctness

    test_ids = test_dataloader.dataset.examples if hasattr(test_dataloader.dataset, "examples") else list(range(len(test_label)))

    device = test_label.device

    eval_vanilla = compute_vanilla(eval_probs)
    test_vanilla = compute_vanilla(test_probs)

    method = "mcd"

    eval_mc_logits = load_samples(method, ind = "eval")  # Load MCD samples if they exist

    if eval_mc_logits is None:
        print("error")
            
    #eval_sws = compute_sws(eval_mc_logits)
    eval_pv = compute_pv(eval_mc_logits)
    #eval_bald = compute_bald(eval_mc_logits)

    test_mc_logits = load_samples(method, ind = "test")  # Load MCD samples if they exist

    if test_mc_logits is None:
        print("error")


    #test_sws = compute_sws(test_mc_logits)
    test_pv = compute_pv(test_mc_logits)
    #test_bald = compute_bald(test_mc_logits)

    # Coverages to evaluate
    coverages = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)]

    # SR/MSP — exact test coverages, CSV outputs
    """sr_metrics = run_sr_baseline_testcal_csv(
    test_probs=test_probs,      # [N_test, C]
    y_test=test_label,          # [N_test]
    ids_test=test_ids,
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "sr_testcal")
)

    # MC-Dropout PV — exact test coverages, CSV outputs
    mcd_metrics = run_mcd_pv_baseline_testcal_csv(
    test_pv=test_pv,            # [N_test]
    test_probs=test_probs,      # [N_test, C] (for preds + ECE)
    y_test=test_label,          # [N_test]
    ids_test=test_ids,
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "mcd_pv_testcal")
)

    # Fit on eval, apply on test
    msp_platt_model = fit_platt_on_msp(eval_probs, eval_label)
    test_probs_ps_msp = apply_platt_msp(msp_platt_model, test_probs)

# Evaluate exactly like SR/MCD; saves CSVs under ps_msp_testcal/
    ps_msp_metrics = run_sr_baseline_testcal_csv(
    test_probs=test_probs_ps_msp,
    y_test=test_label,
    ids_test=test_ids,
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "ps_msp_testcal"),
)
    print("MSP-Platt metrics saved to:", os.path.join(args.output_dir, "ps_msp_testcal"))"""

    # Common coverage grid
    coverages = [round(x, 2) for x in np.arange(0.10, 1.01, 0.10)]

# 1) SR/MSP
    sr_out = run_sr_baseline_testcal_csv(
    test_probs=test_probs,      # [N_test, C] probs (task classes only)
    y_test=test_label,          # [N_test]
    ids_test=test_ids,          # list/array of ids (optional; enables prediction files)
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "sr_testcal")
)

# 2) MCD-PV
    mcd_out = run_mcd_pv_baseline_testcal_csv(
    test_pv=test_pv,            # [N_test] predictive variance (scalar per sample)
    test_probs=test_probs,      # [N_test, C] probs (for yhat & ECE)
    y_test=test_label,          # [N_test]
    ids_test=test_ids,
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "mcd_pv_testcal")
)

# 3) MSP-Platt (fit on eval, apply on test)
    ps_out = run_platt_msp_baseline_testcal_csv(
    eval_probs=eval_probs,      # [N_eval, C_task or C_task+1 if abstain] probs
    eval_labels=eval_label,     # [N_eval]
    test_probs=test_probs,      # [N_test, C_task or C_task+1 if abstain] probs
    test_labels=test_label,     # [N_test]
    ids_test=test_ids,
    coverages=coverages,
    save_dir=os.path.join(args.output_dir, "ps_msp_testcal"),
    exclude_last_class=True,    # set False if you DON'T have an abstain column
    base_name="ps_msp_testcal"
)


if __name__== '__main__':
    main()