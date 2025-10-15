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
#from scaling import *
#from platt_scaling import *

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
        with accelerator.main_process_first():
            train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_model(model, train_dataset,tokenizer,args, accelerator)

        if accelerator.is_main_process:
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)

        trained_model_paths.append(model_dir)

        """# ✅ cleanup to really free GPU memory before next loop
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        accelerator.free_memory() """

    return trained_model_paths

    

def train_model(model, train_dataset, tokenizer, args, accelerator):
    #args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = 1

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.max_steps * args.warmup_ratio) if args.warmup_steps == 0 else args.warmup_steps,
        num_training_steps=args.max_steps
    )

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, scheduler
    )

    total_batch_size = args.train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    model.zero_grad()
    for epoch in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(bar):
            inputs, labels = batch
            with accelerator.accumulate(model):
                loss, logits = model(input_ids=inputs, labels=labels)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # metric gathering
            logits = accelerator.gather_for_metrics(logits)
            labels = accelerator.gather_for_metrics(labels)

        # save only once
    if accelerator.is_main_process:
            model_path = os.path.join(args.output_dir, "model.pth")
            torch.save(accelerator.unwrap_model(model).state_dict(), model_path)
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


    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)


    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 batch_size=args.eval_batch_size, num_workers=8, pin_memory=True)
    


    logger.info("Training/evaluation parameters %s", args)
    method = "ensemble"

    ROOT_DIR = "uncertainty_calibration"

    trained_model_paths = train_deep_ensemble(args, model_class, config, tokenizer, accelerator,num_models=5)
    models = load_ensemble_models(model_class, config, tokenizer, trained_model_paths, args)
    # Get Deep Ensemble logit samples
    eval_ensemble_logits = deep_ensemble_predictions(models, eval_dataloader)  # Shape: (num_models, batch_size, num_classes)
    save_samples(method, eval_ensemble_logits, ind = "eval")



    models = load_ensemble_models(model_class, config, tokenizer, trained_model_paths, args)
    # Get Deep Ensemble logit samples
    test_ensemble_logits = deep_ensemble_predictions(models, test_dataloader)  # Shape: (num_models, batch_size, num_classes)
    save_samples(method, test_ensemble_logits, ind = "test")


if __name__ == '__main__':
    main()

