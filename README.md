# Beyond Confidence: Rethinking Uncertainty Calibration in Deep Code Models

Currently, we support two software engineering tasks: Vulnerability detection and defect prediction. For each folder, we include the code for four models: StarCoder-3B, Qwen2.5-Coder-7B, CodeLlama-7B, DeepSeek-Coder-7B  

### We explain how to run our code in the below:
If you want to run the vulnerability detection task, please go to vulnerability detection folder, then select the model folder you want to run. For each folder, unzip the dataset.zip file. 
Then run: 
python preprocess.py 
file to extract the train, valid, and test dataset. This step can be ignored for defect prediction task. 

For selected model on each task: 

## 1) Finetune the model: 
./run.sh 
For each model, please change the following commands accordingly:
--model_type={model} \
--tokenizer_name={path to model} \
--model_name_or_path={path to model}

## 2) Test your finetuned model: 
./test.sh
For each model, please change the following commands accordingly:
--model_type={model} \
--tokenizer_name={path to model} \
--model_name_or_path={path to model}

## 3) To run uncertainty calibration: 
./run_ts.sh 
For each model, please change the following commands accordingly:
--model_type={model} \
--tokenizer_name={path to model} \
--model_name_or_path={path to model}
This will create different plots, reliability diagrams, as well as evaluation metrics for different uncertainty metrics on each scaling technique. 

## 4) To run behavioral calibration proposed by us: 
./run_ts.sh 
For each model, please change the following commands accordingly:
python behavioral.py \
--model_type={model} \
--tokenizer_name={path to model} \
--model_name_or_path={path to model}
This will create different plots, reliability diagrams, as well as evaluation metrics for different uncertainty metrics on each scaling technique. 

## MECE and MUCE scores for each subject
<details>
  <summary>MECE/MUCE scores</summary>
  Due to README space limits, see the full table here:
  <a href="https://github.com/CODEAIResearch/CODECalibration/blob/main/MECE_MUCE_SCORES/uncertainty_metrics.csv">uncertainty_metrics.csv</a>.
</details>













