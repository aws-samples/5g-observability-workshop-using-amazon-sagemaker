import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import argparse
import pathlib
import time

import boto3
import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--feature-group-name", type=str)
parser.add_argument("--bucket-name", type=str)
parser.add_argument("--bucket-prefix", type=str)
parser.add_argument("--region", type=str, default="us-east-2")
args = parser.parse_args()

region = args.region
boto3.setup_default_session(region_name=region)
s3_client = boto3.client("s3")
account_id = boto3.client("sts").get_caller_identity()["Account"]
now = pd.to_datetime("now")

feature_store_session = sagemaker.Session()
feature_group = FeatureGroup(name=args.feature_group_name, sagemaker_session=feature_store_session)

table_name = (
    feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"]

)

print(f'table_name: {table_name}')

table_prefix = table_name.replace("_", "-")

print(f'table_prefix: {table_prefix}')

feature_group_s3_prefix = f'{args.bucket_prefix}/{account_id}/sagemaker/{region}/offline-store/{table_prefix}/data/year={now.year}/month={now.strftime("%m")}/day={now.strftime("%d")}'

print(f'feature_group_s3_prefix: {feature_group_s3_prefix}')

# wait for data to be added to offline feature store
offline_store_contents = None
while offline_store_contents is None:
    objects = s3_client.list_objects(
        Bucket=args.bucket_name, Prefix=feature_group_s3_prefix
    )

    if "Contents" in objects:
        num_datasets = len(objects["Contents"])
    else:
        num_datasets = 0

    if num_datasets >= 1:
        offline_store_contents = objects["Contents"]
    else:
        print(
            f"Waiting for data in offline store: {args.bucket_name}/{feature_group_s3_prefix}"
        )
        time.sleep(60)

print("Data available.")


query = feature_group.athena_query()

query_string = f"""
SELECT * FROM "{table_name}"
"""

print(query_string)

query.run(query_string=query_string, output_location=f"s3://{args.bucket_name}/query_results/")
query.wait()

dataset = query.as_dataframe()

col_order = ["anomaly"] + list(dataset.drop(["location_id", "anomaly", "eventtime", "write_time","api_invocation_time",'is_deleted'], axis=1).columns)

train = dataset.sample(frac=0.80, random_state=0)[col_order]
test = dataset.drop(train.index)[col_order]

# Write train, test splits to output path
train_output_path = pathlib.Path("/opt/ml/processing/output/train")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
train.to_csv(train_output_path / "train.csv", index=False)
test.to_csv(test_output_path / "test.csv", index=False)
