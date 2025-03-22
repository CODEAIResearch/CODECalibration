# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score,brier_score_loss, auc

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def read_uncertain(filename):
    uncertainty={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,sco = line.split()
            uncertainty[int(idx)] = float(sco)
    return uncertainty

def calculate_scores(answers,predictions):
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==predictions[key])

    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    
    """least = "./saved_models/hid_least.txt"
    ratio = "./saved_models/hid_ratio.txt"
    margin = "./saved_models/hid_margin.txt"
    entropy = "./saved_models/hid_entropy.txt"
    ImportanceLinear = "./saved_models/importance_linear_layer.txt"
    ImportanceLog = "./saved_models/importance_log_layer.txt"
    ImportanceExp = "./saved_models/importance_exp_layer.txt"
    DissectorLinear = "./saved_models/dissector_linear.txt"
    DissectorLog = "./saved_models/dissector_log.txt"
    DissectoExp = "./saved_models/dissector_exp.txt"
    mutual = "./saved_models/hidden_mutual.txt"
    predictive = "./saved_models/hid_predictiveEntropy.txt"
    vanilla = "./saved_models/hid_vanilla.txt"
    temp = "./saved_models/hid_temperature.txt"
    dsmg = "./saved_models/DSMGdropout.txt"""

    dsmg = "../code/saved_models/before_uncertainty.txt"

    prediction_path = "../code/saved_models/predictions.txt"
    ground_truth = "../dataset/test.jsonl"


    args = parser.parse_args()
    answers=read_answers(ground_truth)
    predictions=read_predictions(prediction_path)

    uncertainty_least = read_uncertain(dsmg)
    least_values = list(uncertainty_least.values())

    """uncertainty_least = read_uncertain(least)
    uncertainty_ratio = read_uncertain(ratio)
    uncertainty_margin = read_uncertain(margin)
    uncertainty_entropy = read_uncertain(entropy)
    uncertainty_IL = read_uncertain(ImportanceLinear)
    uncertainty_ILog = read_uncertain(ImportanceLog)
    uncertainty_IE = read_uncertain(ImportanceExp)
    uncertainty_DL = read_uncertain(DissectorLinear)
    uncertainty_DLog = read_uncertain(DissectorLog)
    uncertainty_DE = read_uncertain(DissectoExp)
    uncertainty_mutual = read_uncertain(mutual)
    uncertainty_predictive = read_uncertain(predictive)
    uncertainty_vanilla = read_uncertain(vanilla)
    uncertainty_temp = read_uncertain(temp)
    uncertainty_dsmg = read_uncertain(dsmg)"""
    
    """least_values = list(uncertainty_least.values())
    ratio_values = list(uncertainty_ratio.values())
    margin_values = list(uncertainty_margin.values())
    entropy_values = list(uncertainty_entropy.values())
    ImpL_values = list(uncertainty_IL.values())
    ImpLog_values = list(uncertainty_ILog.values())
    ImpE_values = list(uncertainty_IE.values())
    DisL_values = list(uncertainty_DL.values())
    DisLog_values = list(uncertainty_DLog.values())
    DisE_values = list(uncertainty_DE.values())
    mutual_values = list(uncertainty_mutual.values())
    predictive_values = list(uncertainty_predictive.values())
    vanilla_values = list(uncertainty_vanilla.values())
    temp_values = list(uncertainty_temp.values())
    dsmg_values = list(uncertainty_dsmg.values())"""

    Acc = []
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==predictions[key])
    scores=calculate_scores(answers,predictions)

    y_labels = list(answers.values())
    print("Auc of least",roc_auc_score(Acc, least_values))
    """print("Auc of least",roc_auc_score(Acc, least_values))
    print("Auc of ratio", roc_auc_score(Acc, ratio_values))
    print("Auc of margin", roc_auc_score(Acc, margin_values))
    print("Auc of entropy", roc_auc_score(Acc, entropy_values))
    print("Auc of IL", roc_auc_score(Acc, ImpL_values))
    print("Auc of ILog", roc_auc_score(Acc, ImpLog_values))
    print("Auc of IE", roc_auc_score(Acc, ImpE_values))
    print("Auc of DL", roc_auc_score(Acc, DisL_values))
    print("Auc of DLog", roc_auc_score(Acc, DisLog_values))
    print("Auc of DE", roc_auc_score(Acc, DisE_values))
    print("Auc of mutual", roc_auc_score(Acc, mutual_values))
    print("Auc of predicitive", roc_auc_score(Acc, predictive_values))
    print("Auc of vanilla", roc_auc_score(Acc, vanilla_values))
    print("Auc of temp", roc_auc_score(Acc, temp_values))
    print("Auc of dsmg", roc_auc_score(Acc, dsmg_values))"""
    

    #auROC_Score = roc_auc_score(y_labels, y_scores)
    
    print(scores)

if __name__ == '__main__':
    main()