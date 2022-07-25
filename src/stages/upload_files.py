from google.cloud import storage
import os
import tqdm
import pandas as pd
from datetime import datetime
import sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config


def upload_folder(bucket, cloud_output_folder: str, this_run_folder_name: str, local_folder_path: str):

    folder_base = os.path.basename(local_folder_path)
    for root, _, filenames in os.walk(local_folder_path):
        for filename in filenames:
            relative_root = root.split(folder_base)
            relative_root = relative_root[1] if len(relative_root) > 1 else ""
            destination_blob_name = os.path.join( cloud_output_folder, this_run_folder_name, folder_base, relative_root, filename )
            destination_blob_name = destination_blob_name.replace("\\", "/")

            print("Cloud blob path", destination_blob_name)
            blob = bucket.blob(destination_blob_name)
            source_file_name = os.path.join(root, filename)
            blob.upload_from_filename(source_file_name)



if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(project_root, config.cloud_storage.credentials)
    storage_client = storage.Client(config.cloud_storage.project_name)
    bucket = storage_client.bucket(config.cloud_storage.bucket_name)

    # output/tensorboard/latest/**
    # output/model/**
    new_folder_name = f"{config.trainer.experiment_name}_{datetime.now().strftime('%Y_%m_%d__%H_%M')}"

    local_model_folder = os.path.join(project_root, config.trainer.output_folder, "model")
    local_logs_folder = os.path.join(project_root, config.trainer.tensorboard_log)
    upload_folder(bucket, config.cloud_storage.output_folder, new_folder_name, local_model_folder)
    upload_folder(bucket, config.cloud_storage.output_folder, new_folder_name, local_logs_folder)




