import os
import shutil
import tempfile
from unittest import TestCase

import onnx
import onnxruntime as ort
import torch.nn
from onnxruntime.quantization import quantize

from src.core.model import create_score_model
from src.stages.export_model_to_onnx import create_dummy_input, export_onnx


class Test(TestCase):
    def create_dummy_model(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 3)
                self.conv2 = torch.nn.Conv2d(4, 8, 3)
                self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(8, 64)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor):
                output = self.conv1(x)
                self.relu(output)
                output = self.conv2(output)
                self.relu(output)
                output = self.pooling(output)
                output = torch.flatten(output, 1)
                output = self.fc(output)
                return output

        return DummyModel()

    def test_create_dummy_input(self):
        image = create_dummy_input(224)
        self.assertEqual(image.shape[0], 1)
        self.assertEqual(image.shape[1], 3)
        self.assertEqual(image.shape[2], 224)
        self.assertEqual(image.shape[3], 224)

    def test_export_onnx(self):
        model = self.create_dummy_model()
        temp_folder = tempfile.mkdtemp()
        image = create_dummy_input(224)
        try:
            onnx_file_path = os.path.join(temp_folder, "model.onnx")
            export_onnx(model, 224, onnx_file_path)
            # imported_model = onnx.load(onnx_file_path)
            ort_sess = ort.InferenceSession(onnx_file_path)
            actual_outputs = ort_sess.run(None, {"image": image.numpy()})[0]
            expected_output = model(image).detach().numpy()
            print("Expected", expected_output.flatten().tolist())
            print("ONNX Actual", actual_outputs.flatten().tolist())
            for actual, expected in zip(
                actual_outputs.flatten().tolist(), expected_output.flatten().tolist()
            ):
                self.assertAlmostEqual(actual, expected, delta=0.001)

        finally:
            shutil.rmtree(temp_folder)
