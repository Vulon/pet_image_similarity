import os
import sys

import dvc.api
import pandas as pd
import tqdm
from google.cloud import storage


def _get_storage_blobs(
    project_root: str,
    credentials_path: str,
    bucket_name: str,
    cloud_images_folder: str,
    project_name: str,
) -> tuple[list[storage.Blob], storage.Client]:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        project_root, credentials_path
    )
    storage_client = storage.Client(project_name)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=cloud_images_folder + "/")
    return blobs, storage_client


def _parse_blob_path(path: str) -> tuple[str, str]:
    image_name = os.path.basename(path)
    image_class = os.path.basename(os.path.dirname(path))
    return image_class, image_name


def _create_paths_dataframe(blobs: list[storage.Blob]) -> pd.DataFrame:
    df = []
    for blob in blobs:
        image_class, image_name = _parse_blob_path(blob.name)
        df.append((os.path.join(image_class, image_name), image_class))
    df = pd.DataFrame(df, columns=["path", "class"])
    return df


def _download_images(blobs: list[storage.Blob], images_folder: str):
    for blob in tqdm.tqdm(blobs):
        image_class, image_name = _parse_blob_path(blob.name)
        os.makedirs(os.path.join(images_folder, image_class), exist_ok=True)
        blob.download_to_filename(os.path.join(images_folder, image_class, image_name))


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    sys.path.append(project_root)
    from src.config import get_config_from_dvc

    config = get_config_from_dvc()
    images_folder = os.path.join(project_root, config.data.local_images_folder)
    blobs, storage_client = _get_storage_blobs(
        project_root,
        config.cloud_storage.credentials,
        config.cloud_storage.bucket_name,
        config.cloud_storage.images_folder,
        config.cloud_storage.project_name,
    )
    blobs = list(blobs)
    df = _create_paths_dataframe(blobs)
    _download_images(blobs, os.path.join(project_root, config.data.local_images_folder))

    os.makedirs(
        os.path.join(project_root, os.path.dirname(config.data.classes_text_file)),
        exist_ok=True,
    )
    df.to_csv(os.path.join(project_root, config.data.classes_text_file), index=False)
    storage_client.close()
