from google.cloud import storage
import os
import tqdm
import pandas as pd
import sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config


def _get_storage_blobs(project_root:str, credentials_path: str, bucket_name: str, cloud_images_folder: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(project_root, credentials_path)
    storage_client = storage.Client("nlp-masters-project")
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=cloud_images_folder + "/")
    return blobs, storage_client

def _download_images(blobs,images_folder: str) -> pd.DataFrame:
    df = []
    for blob in tqdm.tqdm(blobs):
        image_name = os.path.basename(blob.name)
        image_class = blob.name.split("/")[1]
        df.append((os.path.join(image_class, image_name), image_class))
        os.makedirs(os.path.join(images_folder, image_class), exist_ok=True)
        blob.download_to_filename(os.path.join(images_folder, image_class, image_name))

    df = pd.DataFrame(df, columns=["path", "class"])
    return df


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)
    images_folder = os.path.join(project_root, config.data.local_images_folder)
    blobs, storage_client = _get_storage_blobs(project_root, config.cloud_storage.credentials, config.cloud_storage.bucket_name, config.cloud_storage.images_folder)
    df = _download_images(blobs, os.path.join(project_root, config.data.local_images_folder))

    os.makedirs( os.path.join(project_root, os.path.dirname(config.data.classes_text_file)), exist_ok=True )
    df.to_csv(os.path.join(project_root, config.data.classes_text_file))
    storage_client.close()