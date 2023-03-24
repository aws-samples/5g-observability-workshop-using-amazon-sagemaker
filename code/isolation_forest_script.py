from __future__ import print_function

import argparse
import os
from io import StringIO

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sagemaker.session import Session
from sagemaker.experiments.run import load_run
import boto3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--region", type=str, default="us-east-2", help="SageMaker Region")
    
    args = parser.parse_args()
        
    session = Session(boto3.session.Session(region_name=args.region))
    with load_run(sagemaker_session=session) as run:
    
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
            print(f)

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]

        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
                ).format(args.train, "train")
            )

        raw_data = [pd.read_csv(file, engine="python") for file in input_files]
        train_data = pd.concat(raw_data)
        
        run.log_parameter("num_train_samples",  len(train_data.index))


        input_data = train_data.values

        clf = IsolationForest(max_samples=args.max_samples, random_state=args.random_state)
        clf.fit(input_data)

        result_df = pd.DataFrame(input_data)
        # compute metrics
        y_pred_train = clf.predict(input_data)
        y_score_train = clf.score_samples(input_data)

        result_df['prediction'] = y_pred_train
        result_df['score'] = y_score_train

        anomalies = result_df[result_df['prediction']==-1]['prediction'].count()
        total = result_df.shape[0]

        score_mean = result_df["score"].mean()
        score_std = result_df["score"].std()

        print(f"#_anomalies = {anomalies}; pct_anomalies = {anomalies/total*100} %;")

        print(f"avg_score = {score_mean:.2f}; std = {score_std: .2f};")
        
        run.log_metric(name="#_anomalies", value=anomalies)
        run.log_metric(name="pct_anomalies", value=anomalies/total)
        
        run.log_metric(name="avg_score", value=score_mean)
        run.log_metric(name="std", value=score_std)

        # Print the coefficients of the trained classifier, and save the coefficients
        joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
        run.log_file(os.path.join(args.model_dir, "model.joblib"), 
                     name="model_output", is_output=True)
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf