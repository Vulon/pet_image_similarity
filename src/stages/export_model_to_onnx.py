import torch
import onnx


from src.config import get_config
from src.core.model import ResNetScore


config = get_config(None, yaml_filepath="C:/PythonProjects/image_similarity/params.yaml")

loss_function = torch.nn.MSELoss()

dummy_input = torch.randn(1, 3, config.augmentations.image_size, config.augmentations.image_size, device="cpu")
model = ResNetScore(config.model.output_vector_size)

torch.onnx.export(model, dummy_input, "resnet.onnx", verbose=True, input_names=["image"], output_names=["vector"])

imported_model = onnx.load("resnet.onnx")

onnx.checker.check_model(imported_model)