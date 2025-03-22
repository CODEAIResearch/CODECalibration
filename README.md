# Beyond Confidence: Rethinking Uncertainty Calibration in Deep Code Models

Currently, we support two software engineering tasks: Vulnerability detection and defect prediction. For each folder, we include the code for three models: CodeBERT, GraphCodeBERT, and RoBERTa. 

### We explain how to run our code in the below:
If you want to run the vulnerability detection task, please go to vulnerability detection folder, then select the model folder you want to run. For each folder, unzip the dataset.zip file. 
Then run: 
python preprocess.py 
file to extract the train, valid, and test dataset. This step can be ignored for defect prediction task. 

For selected model on each task: 

## 1) Finetune the model: 
./run.sh 

## 2) Test your finetuned model: 
./test.sh

## 3) To generate sub-models: 
./emsemble.sh

## 4) To run uncertainty calibration: 
./run_ts.sh 
This will create different plots, reliability diagrams, as well as evaluation metrics for different uncertainty metrics on each scaling technique. 













