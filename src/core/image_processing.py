from imgaug import augmenters as iaa

from src.config import AugmentationsConfig


def create_crop_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    crop_sequence = iaa.Sequential(
        [
            iaa.Resize({"shorter-side": config.image_size, "longer-side": "keep-aspect-ratio"}),
            iaa.CenterCropToSquare()
        ]
    )
    return crop_sequence


def _create_rotate_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    rotate_sequence = iaa.Sequential([
        iaa.SomeOf(1, [
            iaa.Fliplr(),
            iaa.Rotate((-config.rotate_abs_angle, config.rotate_abs_angle), mode="reflect")
        ])
    ])
    return rotate_sequence


def _create_arithmetic_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    arithmetic_sequence = iaa.Sequential([
        iaa.SomeOf(1,
           [
               iaa.AdditiveGaussianNoise(scale=(config.noise_lower_bound, config.noise_upper_bound)),
               iaa.AdditiveLaplaceNoise(scale=(config.noise_lower_bound, config.noise_upper_bound)),
               iaa.AdditivePoissonNoise(lam=(config.noise_lower_bound, config.noise_upper_bound)),
               iaa.SaltAndPepper(p=config.salt_and_pepper_p),
               iaa.Dropout(config.dropout_p),
               iaa.JpegCompression(compression=config.jpeg_compression_bounds)
           ]
        )
    ])
    return arithmetic_sequence


def _create_blur_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    blur_sequence = iaa.Sequential([
        iaa.SomeOf(1,
           [
               iaa.AverageBlur((config.blur_lower_kernel_bound, config.blur_upper_kernel_bound)),
               iaa.BilateralBlur((config.blur_lower_kernel_bound, config.blur_upper_kernel_bound)),
               iaa.MedianBlur((config.blur_lower_kernel_bound, config.blur_upper_kernel_bound)),
               iaa.MotionBlur(config.motion_blur_bounds),
               iaa.GaussianBlur(config.gaussian_blur_bounds),
           ]
        )
    ])
    return blur_sequence


def _create_color_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    color_sequence = iaa.Sequential([
        iaa.SomeOf(1,
           [
               iaa.MultiplyAndAddToBrightness(mul=config.color_multiplier, add=(-30, 30)),
               iaa.MultiplyHueAndSaturation(mul=config.color_multiplier),
           ]
        )
    ])
    return color_sequence


def _create_contrast_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    contrast_sequence = iaa.Sequential([
        iaa.SomeOf(1,
           [
               iaa.GammaContrast(config.contrast_bounds),
               iaa.CLAHE(clip_limit=config.clahe_clip_limit),
               iaa.LinearContrast(config.contrast_bounds)
           ]
        )
    ])
    return contrast_sequence


def create_train_sequence(config: AugmentationsConfig) -> iaa.Augmenter:
    train_sequence = iaa.Sequential([
        iaa.Sometimes(config.rotate_probability, _create_rotate_sequence(config)),
        iaa.Sometimes(config.color_probability, _create_color_sequence(config)),
        iaa.Sometimes(config.contrast_probability, _create_contrast_sequence(config)),
        iaa.Sometimes(config.arithmetic_probability, _create_arithmetic_sequence(config)),
        iaa.Sometimes(config.blur_probability, _create_blur_sequence(config)),
    ])
    return train_sequence
