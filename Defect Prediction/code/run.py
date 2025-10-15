from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json


from tqdm import tqdm, trange
import multiprocessing
from model import DecoderClassifier

cpu_cont = multiprocessing.cpu_count()


from torch.optim import AdamW
from transformers import BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, get_scheduler,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer, 
                          AutoConfig, AutoModel, AutoTokenizer)



import logging

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'codellama' : (AutoConfig,AutoModel, AutoTokenizer),
    'deepseek' : (AutoConfig,AutoModel, AutoTokenizer),
    'codegemma' : (AutoConfig,AutoModel, AutoTokenizer)
    }

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

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
    elif args.model_type in ["starcoder", "deepseek"]:
        code_tokens=tokenizer.tokenize(code)
        source_tokens = code_tokens[:args.block_size]
    else:
        code_tokens=tokenizer.tokenize(code)
        code_tokens = code_tokens[:args.block_size-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    if args.model_type in ["codellama"]:
        source_ids = tokenizer.encode(js['input'].split("</s>")[0], max_length=args.block_size, padding='max_length', truncation=True)
    else:
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


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)



def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Single-GPU training loop (frozen backbone + linear head). """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    args.eval_batch_size = max(1, args.eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False,
                                 batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / max(1, args.gradient_accumulation_steps))
    args.num_train_epochs = int(args.epoch)
    args.max_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Optimizer: only trainable params (your head)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Scheduler
    warmup_steps = int(args.max_steps * args.warmup_ratio) if args.warmup_steps == 0 else args.warmup_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=args.max_steps)

    # Logs
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per GPU = {args.train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_steps}")

    best_f1 = -1.0
    best_acc = -1.0
    patience = 0
    global_step = 0

    model.zero_grad(set_to_none=True)

    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()  # only the head has requires_grad=True
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader))
        for step, batch in enumerate(pbar):
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward
            loss, logits = model(input_ids=inputs, labels=labels)
            loss = loss / max(1, args.gradient_accumulation_steps)
            loss.backward()

            # Step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            pbar.set_postfix(loss=f"{running_loss/max(1,seen):.4f}")

            global_step += 1

        # ----- End epoch: evaluate -----
        results = evaluate(args, model, eval_dataloader, eval_dataset,device)
        print({k: round(v, 4) for k, v in results.items()})

        # ----- Save checkpoints on improvements -----
        improved_f1 = results["eval_f1"] > best_f1
        improved_acc = results["eval_acc"] > best_acc

        if improved_f1:
            best_f1 = results["eval_f1"]
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-best-f1/{args.project}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.bin"))
            print(f"[Saved] best F1 @ epoch {epoch} to {ckpt_dir}")

        if improved_acc:
            best_acc = results["eval_acc"]
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-best-acc/{args.project}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.bin"))
            print(f"[Saved] best Acc @ epoch {epoch} to {ckpt_dir}")

        if improved_f1 or improved_acc:
            patience = 0
        else:
            patience += 1

        if args.max_patience > 0 and patience >= args.max_patience:
            print(f"Early stopping: patience {patience} (best_f1={best_f1:.2f}, best_acc={best_acc:.2f})")
            # ensure at least one checkpoint exists
            if best_f1 < 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-best-f1/{args.project}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.bin"))
            break

    # Optional test pass using your existing test() (adapt it to no-accelerate)
    if getattr(args, "do_test", False):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-best-f1/{args.project}", "model.bin")
        if os.path.isfile(ckpt_dir):
            state = torch.load(ckpt_dir, map_location=device)
            model.load_state_dict(state, strict=False)
            model.to(device)
        result = test(args, model, tokenizer, device)  # adapt your test() to no-accelerate
        print("***** Test results *****")
        for k in sorted(result.keys()):
            print(f"  {k} = {round(result[k], 4)}")             

def calculate_metrics(labels, preds, average='macro'):
    acc=accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average=average, zero_division=0)
    recall = recall_score(labels, preds,average=average, zero_division=0)
    f1 = f1_score(labels, preds,average=average, zero_division=0)
    """TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(TP+FN)"""
    cm = confusion_matrix(labels, preds)
    TP = np.diag(cm).sum()
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (TP + FP.sum() + FN.sum())

    # Aggregate version
    tnr = TN / (TN + FP.sum()) if (TN + FP.sum()) > 0 else 0
    fpr = FP.sum() / (FP.sum() + TN) if (FP.sum() + TN) > 0 else 0
    fnr = FN.sum() / (TP + FN.sum()) if (TP + FN.sum()) > 0 else 0
    return round(acc,4)*100, round(prec,4)*100, \
        round(recall,4)*100, round(f1,4)*100, round(tnr,4)*100, \
            round(fpr,4)*100, round(fnr,4)*100

def evaluate(args, model, eval_dataloader, eval_dataset, device,eval_when_training: bool=False):
    """
    Single-GPU evaluation. No accelerate/DDP.
    """
    # Ensure output dir exists (optional)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir, exist_ok=True)

    # Logging
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size  = %d", args.eval_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)
    model.eval()

    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            loss, logits = model(input_ids=inputs, labels=labels)
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    eval_loss = float(np.mean(losses)) if losses else 0.0
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, args.num_labels))
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=int)

    preds = logits_np.argmax(axis=1) if logits_np.size else np.array([], dtype=int)
    eval_acc, eval_prec, eval_recall, eval_f1, eval_tnr, eval_fpr, eval_fnr = calculate_metrics(labels_np, preds)

    result = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_prec": eval_prec,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_tnr": eval_tnr,
        "eval_fpr": eval_fpr,
        "eval_fnr": eval_fnr,
    }
    return result


def test(args, model, tokenizer, device=None):
    """
    Single-GPU test. No accelerate/DDP.
    Builds its own dataloader from args.test_data_file using your TextDataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    # Build dataset/dataloader
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    args.eval_batch_size = max(1, args.eval_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size  = %d", args.eval_batch_size)

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Testing"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward without labels → returns logits
            logits = model(input_ids=inputs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, args.num_labels))
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=int)
    preds = logits_np.argmax(axis=1) if logits_np.size else np.array([], dtype=int)

    test_acc, test_prec, test_recall, test_f1, test_tnr, test_fpr, test_fnr = calculate_metrics(labels_np, preds)

    result = {
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_tnr": test_tnr,
        "test_fpr": test_fpr,
        "test_fnr": test_fnr,
    }
    return result
    
    
def test_prob_no_dist(args, model, tokenizer):
    """
    Writes predictions.txt with class labels (one per line: "<idx>\\t<pred>").
    Mirrors your original behavior but without accelerate/DataParallel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #model.to(device)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    loader = DataLoader(eval_dataset, shuffle=False, batch_size=max(1, args.eval_batch_size), num_workers=4, pin_memory=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test(prob)"):
            inputs, labels = batch
            inputs = inputs.to(device)
            logits = model(input_ids=inputs)          # forward without labels -> logits [B,C]
            pred = logits.argmax(dim=1).cpu().numpy() # class ids
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    os.makedirs(os.path.join(args.output_dir, args.project), exist_ok=True)
    out_path = os.path.join(args.output_dir, args.project, "predictions.txt")
    with open(out_path, "w") as f:
        for ex, p in zip(eval_dataset.examples, preds):
            f.write(f"{ex.idx}\t{int(p)}\n")
    logger.info(f"Saved predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--project', type=str, required=True, help="using dataset from this project.")
    parser.add_argument('--train_project', type=str, required=False, help="using training dataset from this project.")
    #parser.add_argument('--model_dir', type=str, required=True, help="directory to store the model weights.")
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--test_cwe', type=str, default=None, required=False, help="using dataset from this CWE for testing.")
    
    # run dir
    parser.add_argument('--run_dir', type=str, default="runs", help="parent directory to store run stats.")

    ## Other parameters
    parser.add_argument("--max_source_length", default=400, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="codegen", type=str,
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
    parser.add_argument("--do_test_prob", action='store_true',
                        help="Whether to run eval and save the prediciton probabilities.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--weighted_sampler", action='store_true',
                        help="Whether to do project balanced sampler using WeightedRandomSampler.")
    # Soft F1 loss function
    parser.add_argument("--soft_f1", action='store_true',
                        help="Use soft f1 loss instead of regular cross entropy loss.")
    parser.add_argument("--class_weight", action='store_true',
                        help="Use class weight in the regular cross entropy loss.")
    parser.add_argument("--vul_weight", default=1.0, type=float,
                        help="Weight for the vulnerable class in the regular cross entropy loss.")

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
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup ratio over all steps.")

    parser.add_argument('--logging_steps', type=int, default=1000,
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
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--max-patience', type=int, default=-1, help="Max iterations for model with no improvement.")

    

    args = parser.parse_args()

    # Setup distant debugging if needed
    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    #logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                    datefmt='%m/%d/%Y %H:%M:%S',
    #                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    #logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    #set_seed(args.seed)

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

    #compute_dtype = getattr(torch, "bfloat16")
    #quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

    if args.model_name_or_path:
        if args.model_type in ["starcoder","deepseek"]:
            model = model_class.from_pretrained(args.model_name_or_path,
                                                #use_cache = False,
                                                torch_dtype = torch.bfloat16,trust_remote_code=True )
        else:
            print("load from here")
            model = model_class.from_pretrained(args.model_name_or_path,
                                                torch_dtype = torch.bfloat16, trust_remote_code=True)
    else:
        model = model_class(config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    #config.pad_token_id = tokenizer(tokenizer.pad_token, truncation=True)['input_ids'][0]
    if args.model_type in ['codellama', 'starcoder',"deepseek"]:
        model = DecoderClassifier(model,config,tokenizer,args)
    else:
        model = Model(model,config,tokenizer,args)

    logger.info("Training/evaluation parameters %s", args)
    
    
   
    # Training
    if args.do_train:
        
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

        train(args, train_dataset, eval_dataset, model, tokenizer)


    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = f'checkpoint-best-f1/{args.project}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir), strict=False)      
        model.to(args.device)
    
    if args.do_test_prob and args.local_rank in [-1, 0]:
        """from safetensors.torch import load_file
        checkpoint_path = "saved_models/checkpoint-best-f1/vul/model.safetensors"  # Adjust path
        state_dict = load_file(checkpoint_path)

        # Save as model.bin
        torch.save(state_dict, "saved_models/checkpoint-best-acc/vul/model.bin")"""

        checkpoint_prefix = f'checkpoint-best-acc/{args.project}/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir), strict=False)                  
        model.to(args.device)
        test_prob_no_dist(args, model, tokenizer)
    
    return results


if __name__ == "__main__":
    main()

