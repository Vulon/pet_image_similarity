from unittest import TestCase, TestSuite, TextTestRunner
import os
import sys
import dvc
import tempfile
import shutil
from unittest import mock
from unittest.mock import Mock, patch, MagicMock


class UnitTest(TestCase):
    def test__parse_blob_path(self):
        print("test__parse_blob_path")
        from src.stages.download_images import _parse_blob_path

        image_class, image_name = _parse_blob_path("cat_1/image.png")
        self.assertEqual(image_class, "cat_1")
        self.assertEqual(image_name, "image.png")

        image_class, image_name = _parse_blob_path("some_folder/cat_1/image.jpg")
        self.assertEqual(image_class, "cat_1")
        self.assertEqual(image_name, "image.jpg")

        image_class, image_name = _parse_blob_path("more_folders/another_folder/cat_1/image.png")
        self.assertEqual(image_class, "cat_1")
        self.assertEqual(image_name, "image.png")

        image_class, image_name = _parse_blob_path(os.path.join("folder", "another_folder", "cat_1", "image.png"))
        self.assertEqual(image_class, "cat_1")
        self.assertEqual(image_name, "image.png")


    def test__create_paths_dataframe(self):
        print("test__create_paths_dataframe")
        from src.stages.download_images import _create_paths_dataframe
        blob_names = [
            os.path.join('cat_1', "blob_1"), os.path.join('cat_3', "blob_b"), os.path.join("folder", 'cat_1', "blob_1")
        ]
        blobs = []
        for name in blob_names:
            mock_blob = Mock()
            mock_blob.name = name
            blobs.append(mock_blob)

        df = _create_paths_dataframe(blobs)
        self.assertDictEqual(
            df.to_dict(),
            {
                "path" : {i: item for i, item in enumerate(
                    [ os.path.join("cat_1", "blob_1"), os.path.join("cat_3", "blob_b"), os.path.join("cat_1", "blob_1") ]
                )},
                "class" : {i : item for i, item in enumerate(["cat_1", "cat_3", "cat_1"])}
            }
        )
