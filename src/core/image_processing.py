import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from transformers import AutoFeatureExtractor

from src.config import AugmentationsConfig


def create_feature_extractor(feature_extractor_name: str):
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
    return feature_extractor


def create_crop_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    crop_sequence = iaa.Sequential(
        [
            iaa.Resize(
                {"shorter-side": config.image_size, "longer-side": "keep-aspect-ratio"},
                random_state=random_seed,
            ),
            iaa.CenterCropToSquare(random_state=random_seed),
        ],
        random_state=random_seed,
    )
    return crop_sequence


def _create_rotate_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    rotate_sequence = iaa.Sequential(
        [
            iaa.SomeOf(
                1,
                [
                    iaa.Fliplr(random_state=random_seed),
                    iaa.Rotate(
                        (-config.rotate_abs_angle, config.rotate_abs_angle),
                        mode="reflect",
                        random_state=random_seed,
                    ),
                ],
                random_state=random_seed,
            )
        ],
        random_state=random_seed,
    )
    return rotate_sequence


def _create_arithmetic_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    arithmetic_sequence = iaa.Sequential(
        [
            iaa.SomeOf(
                1,
                [
                    iaa.AdditiveGaussianNoise(
                        scale=(config.noise_lower_bound, config.noise_upper_bound),
                        random_state=random_seed,
                    ),
                    iaa.AdditiveLaplaceNoise(
                        scale=(config.noise_lower_bound, config.noise_upper_bound),
                        random_state=random_seed,
                    ),
                    iaa.AdditivePoissonNoise(
                        lam=(config.noise_lower_bound, config.noise_upper_bound),
                        random_state=random_seed,
                    ),
                    iaa.SaltAndPepper(
                        p=config.salt_and_pepper_p, random_state=random_seed
                    ),
                    iaa.Dropout(config.dropout_p, random_state=random_seed),
                    iaa.JpegCompression(
                        compression=config.jpeg_compression_bounds,
                        random_state=random_seed,
                    ),
                ],
                random_state=random_seed,
            )
        ],
        random_state=random_seed,
    )
    return arithmetic_sequence


def _create_blur_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    blur_sequence = iaa.Sequential(
        [
            iaa.SomeOf(
                1,
                [
                    iaa.AverageBlur(
                        (
                            config.blur_lower_kernel_bound,
                            config.blur_upper_kernel_bound,
                        ),
                        random_state=random_seed,
                    ),
                    iaa.BilateralBlur(
                        (
                            config.blur_lower_kernel_bound,
                            config.blur_upper_kernel_bound,
                        ),
                        random_state=random_seed,
                    ),
                    iaa.MedianBlur(
                        (
                            config.blur_lower_kernel_bound,
                            config.blur_upper_kernel_bound,
                        ),
                        random_state=random_seed,
                    ),
                    iaa.MotionBlur(config.motion_blur_bounds, random_state=random_seed),
                    iaa.GaussianBlur(
                        config.gaussian_blur_bounds, random_state=random_seed
                    ),
                ],
                random_state=random_seed,
            )
        ],
        random_state=random_seed,
    )
    return blur_sequence


def _create_color_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    color_sequence = iaa.Sequential(
        [
            iaa.SomeOf(
                1,
                [
                    iaa.MultiplyAndAddToBrightness(
                        mul=config.color_multiplier,
                        add=(-30, 30),
                        random_state=random_seed,
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul=config.color_multiplier, random_state=random_seed
                    ),
                ],
                random_state=random_seed,
            )
        ],
        random_state=random_seed,
    )
    return color_sequence


def _create_contrast_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    contrast_sequence = iaa.Sequential(
        [
            iaa.SomeOf(
                1,
                [
                    iaa.GammaContrast(config.contrast_bounds, random_state=random_seed),
                    iaa.CLAHE(
                        clip_limit=config.clahe_clip_limit, random_state=random_seed
                    ),
                    iaa.LinearContrast(
                        config.contrast_bounds, random_state=random_seed
                    ),
                ],
                random_state=random_seed,
            )
        ],
        random_state=random_seed,
    )
    return contrast_sequence


def create_train_sequence(
    config: AugmentationsConfig, random_seed: int
) -> iaa.Augmenter:
    train_sequence = iaa.Sequential(
        [
            iaa.Sometimes(
                config.rotate_probability, _create_rotate_sequence(config, random_seed)
            ),
            iaa.Sometimes(
                config.color_probability, _create_color_sequence(config, random_seed)
            ),
            iaa.Sometimes(
                config.contrast_probability,
                _create_contrast_sequence(config, random_seed),
            ),
            iaa.Sometimes(
                config.arithmetic_probability,
                _create_arithmetic_sequence(config, random_seed),
            ),
            iaa.Sometimes(
                config.blur_probability, _create_blur_sequence(config, random_seed)
            ),
        ]
    )
    return train_sequence


def read_image(image_path: str):
    image = Image.open(image_path)
    image = np.uint8(np.array(image.getdata()).reshape((-1, image.width, 3)))
    return image
