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
    train_fracture: float
    val_fracture: float
    test_fracture: float
    train_classes_file: str
    val_classes_file: str
    test_classes_file: str
    train_h5_file: str
    val_h5_file: str
    test_h5_file: str

@dataclass
class AugmentationsConfig:
    image_size: int
    rotate_abs_angle: int
    noise_lower_bound: int
    noise_upper_bound: int
    dropout_p: float
    salt_and_pepper_p: float
    blur_lower_kernel_bound: int
    blur_upper_kernel_bound: int
    jpeg_compression_bounds: tuple[int, int]
    motion_blur_bounds: tuple[int, int]
    gaussian_blur_bounds: tuple[float, float]
    color_multiplier: tuple[float, float]
    contrast_bounds: tuple[float, float]
    clahe_clip_limit: tuple[float, float]

    rotate_probability: float
    arithmetic_probability: float
    blur_probability: float
    color_probability: float
    contrast_probability: float

@dataclass
class BaseConfig:
    random_seed: int
    cloud_storage: CloudStorageConfig
    data: DataConfig
    augmentations: AugmentationsConfig


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
