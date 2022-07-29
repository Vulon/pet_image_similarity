import os
import sys
from datetime import datetime

import dvc.api
import pandas as pd
import tqdm
from google.cloud import storage

sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config_from_dvc


def upload_folder(
    bucket, cloud_output_folder: str, this_run_folder_name: str, local_folder_path: str
):

    folder_base = os.path.basename(local_folder_path)
    for root, _, filenames in os.walk(local_folder_path):
        for filename in filenames:
            relative_root = root.split(folder_base)
            relative_root = relative_root[1] if len(relative_root) > 1 else ""
            destination_blob_name = os.path.join(
                cloud_output_folder,
                this_run_folder_name,
                folder_base,
                relative_root,
                filename,
            )
            destination_blob_name = destination_blob_name.replace("\\", "/")

            print("Cloud blob path", destination_blob_name)
            blob = bucket.blob(destination_blob_name)
            source_file_name = os.path.join(root, filename)
            blob.upload_from_filename(source_file_name)


def upload_file(bucket, full_cloud_folder_path: str, local_file_path: str):
    file_name = os.path.basename(local_file_path)
    destination_blob_name = os.path.join(full_cloud_folder_path, file_name).replace(
        "\\", "/"
    )
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]

    config = get_config_from_dvc()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        project_root, config.cloud_storage.credentials
    )
    storage_client = storage.Client(config.cloud_storage.project_name)
    bucket = storage_client.bucket(config.cloud_storage.bucket_name)

    # output/tensorboard/latest/**
    # output/model/**
    new_folder_name = (
        f"{config.trainer.experiment_name}_{datetime.now().strftime('%Y_%m_%d__%H_%M')}"
    )

    local_model_folder = os.path.join(
        project_root, config.trainer.output_folder, "model"
    )
    local_logs_folder = os.path.join(project_root, config.trainer.tensorboard_log)
    upload_folder(
        bucket, config.cloud_storage.output_folder, new_folder_name, local_model_folder
    )
    upload_folder(
        bucket, config.cloud_storage.output_folder, new_folder_name, local_logs_folder
    )

    local_plot_path = os.path.join(project_root, config.data.visualization_output_path)
    upload_file(
        bucket,
        os.path.join(config.cloud_storage.output_folder, new_folder_name),
        local_plot_path,
    )
    onnx_output_path = os.path.join(
        project_root, config.trainer.output_folder, config.score.onnx_output_filepath
    )
    upload_file(
        bucket,
        os.path.join(config.cloud_storage.output_folder, new_folder_name),
        onnx_output_path,
    )
