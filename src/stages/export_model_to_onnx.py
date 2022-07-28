import os
import shutil
import tempfile

import onnx
import torch
from onnxruntime.quantization import quantize

from src.config import get_config_from_dvc
from src.core.model import create_score_model


def create_dummy_input(image_size: int):
    dummy_input = torch.randn(1, 3, image_size, image_size, device="cpu")
    return dummy_input


def export_onnx(model: torch.nn.Module, input_image_size: int, onnx_output_path: str):
    temp_folder = tempfile.mkdtemp()
    try:
        torch.onnx.export(
            model,
            create_dummy_input(input_image_size),
            os.path.join(temp_folder, "model.onnx"),
            verbose=True,
            input_names=["image"],
            output_names=["vector"],
        )
        quantize.quantize_dynamic(
            os.path.join(temp_folder, "model.onnx"),
            onnx_output_path,
            weight_type=quantize.QuantType.QUInt8,
        )
    finally:
        shutil.rmtree(temp_folder)


if __name__ == "__main__":
    config = get_config_from_dvc()

    loss_function = torch.nn.MSELoss()

    project_root = os.environ["DVC_ROOT"]
    input_model_path = os.path.join(
        project_root, config.trainer.output_folder, "model", "pytorch_model.bin"
    )
    model = create_score_model(
        config.model.pretrained_model_name,
        config.model.output_vector_size,
        input_model_path,
    )

    onnx_output_path = os.path.join(
        project_root, config.trainer.output_folder, config.score.onnx_output_filepath
    )
    export_onnx(model, config.augmentations.image_size, onnx_output_path)

    imported_model = onnx.load(onnx_output_path)

    onnx.checker.check_model(imported_model)
