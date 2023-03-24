import logging

import pandas as pd
import argparse
import pathlib
import json
import os
import numpy as np
import tarfile
import uuid
import pickle
import xgboost as xgb

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_path = "/opt/ml/processing/model" #"model" # 
data_path = '/opt/ml/processing/input/test' #"output" # 

def calibrate(probabilities, cutoff=.2):
    predictions = []
    for p in probabilities:
        if p <= cutoff:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    parser.add_argument("--cutoff-threshold", type=float, default=0.4)
    args, _ = parser.parse_known_args()
    
    print("Extracting the model")

    model_file = os.path.join(model_path, args.model_file)
    file = tarfile.open(model_file)
    file.extractall(model_path)

    file.close()
    
    print("Load xgboost model....")
    with open(os.path.join(model_path, "xgboost-model"), "rb") as f:
        booster = pickle.load(f)
        
    print("Load input data......")
    data = pd.read_csv(os.path.join(data_path, "test.csv"))
    test = data.drop("anomaly", axis=1)
    
    acts = data["anomaly"]
    
    dtest = xgb.DMatrix(test)

    print("Make predictions......")
    probabilities = booster.predict(dtest)
    
    preds = np.asarray(calibrate(probabilities, args.cutoff_threshold))
 
    class_name_list = ['Anomaly', 'Not Anomaly']
    
    precision = precision_score(acts, preds, average='micro')
    recall = recall_score(acts, preds, average='micro')
    accuracy = accuracy_score(acts, preds)
    cnf_matrix = confusion_matrix(acts, preds, labels=range(len(class_name_list)))
    f1 = f1_score(acts, preds, average='micro')
    
    print("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(cnf_matrix))
    logger.debug("F1 score: {}".format(f1))
    
    print(cnf_matrix)
    
    matrix_output = dict()
    
    for i in range(len(cnf_matrix)):
        matrix_row = dict()
        for j in range(len(cnf_matrix[0])):
            matrix_row[j] = int(cnf_matrix[i][j])
        matrix_output[i] = matrix_row

    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "confusion_matrix":matrix_output
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))