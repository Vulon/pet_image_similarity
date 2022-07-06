from dataclasses import dataclass
import dataclasses
import yaml


@dataclass
class CloudStorageConfig:
    credentials: str
    bucket_name: str
    images_folder: str

@dataclass
class DataConfig:
    local_images_folder: str
    classes_text_file: str

@dataclass
class BaseConfig:
    random_seed: int
    cloud_storage: CloudStorageConfig
    data: DataConfig


def __dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in dataclasses.fields(klass)}
        return klass(**{f:__dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:
        return d # Not a dataclass field

def get_config(yaml_dict: dict, yaml_filepath: str = None) -> BaseConfig:
    if yaml_filepath is not None:
        with open(yaml_filepath, 'r') as file:
            yaml_dict = yaml.safe_load(file)
    config = __dataclass_from_dict(BaseConfig, yaml_dict)
    return config
