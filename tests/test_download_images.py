from unittest import TestCase, TestSuite, TextTestRunner
import os
import sys
import dvc
import tempfile
import shutil


class CloudTest(TestCase):

    def tearDown(self):
        if self.storage_client is not None:
            self.storage_client.close()

    def setUp(self):
        self.get_project_root()
        self.get_config()
        self.blobs = None
        self.storage_client = None

    def get_config(self):
        from src.config import get_config
        # params = dvc.api.params_show()
        config = get_config(None, yaml_filepath=os.path.join(self.project_root, "params.yaml"))
        self.config = config

    def get_project_root(self):
        file_path = os.path.abspath(__file__)
        tests_folder = os.path.dirname(file_path)
        project_root = os.path.dirname(tests_folder)
        # sys.path.append(project_root)
        self.project_root = project_root

    def test_project_root(self):
        file_names = os.listdir(self.project_root)
        self.assertIn("src", file_names)
        self.assertIn("dvc.yaml", file_names)
        self.assertIn("params.yaml", file_names)
        self.assertIn("Pipfile", file_names)
        self.assertIn("Pipfile.lock", file_names)


    def get_storage_blobs(self):
        if self.storage_client is None or self.blobs is None:
            from src.stages.download_images import _get_storage_blobs
            config = self.config

            blobs, storage_client = _get_storage_blobs(
                self.project_root,
                config.cloud_storage.credentials,
                config.cloud_storage.bucket_name,
                config.cloud_storage.images_folder,
                config.cloud_storage.project_name,
            )
            self.blobs = [blob for blob in blobs]
            self.storage_client = storage_client
        return self.blobs, self.storage_client

    def test__get_storage_blobs(self):
        blobs, _ = self.get_storage_blobs()
        self.assertGreater(len(blobs), 0)


    def test__download_images(self):
        from src.stages.download_images import _download_images
        temp_folder = tempfile.mkdtemp()
        blobs, _ = self.get_storage_blobs()
        blobs = blobs[: 5]
        _download_images(
            blobs,
            images_folder=temp_folder
        )
        blob_names = [ os.path.basename(b.name) for b in blobs]
        for folder_name in os.listdir(temp_folder):
            images = os.listdir(os.path.join(temp_folder, folder_name))
            self.assertGreater( len(images), 0 )
            for image in images:
                self.assertIn(image, blob_names)
        shutil.rmtree(temp_folder)


def suite():
    suite = TestSuite()
    suite.addTest(CloudTest('test_project_root'))
    suite.addTest(CloudTest('test__get_storage_blobs'))
    suite.addTest(CloudTest('test__download_images'))
    return suite

if __name__ == '__main__':
    runner = TextTestRunner()
    runner.run(suite())