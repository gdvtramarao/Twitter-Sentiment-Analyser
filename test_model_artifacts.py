import importlib
import os
import tempfile
import unittest
from pathlib import Path

import model_artifacts


class ModelArtifactTests(unittest.TestCase):
    def test_paths_stay_relative_to_the_app_outside_the_repository(self):
        original_directory = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)
                artifacts = importlib.reload(model_artifacts)
        finally:
            os.chdir(original_directory)

        repository = Path(__file__).resolve().parent
        self.assertEqual(
            artifacts.MODEL_ARTIFACT_PATH,
            repository / "outputs" / "logreg_model.pkl",
        )
        self.assertEqual(
            artifacts.LABEL_ENCODER_ARTIFACT_PATH,
            repository / "outputs" / "label_encoder.pkl",
        )
        self.assertTrue(artifacts.MODEL_ARTIFACT_PATH.is_file())
        self.assertTrue(artifacts.LABEL_ENCODER_ARTIFACT_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
