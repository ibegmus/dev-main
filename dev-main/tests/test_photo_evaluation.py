import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location(
    "photo_evaluation",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_photo_observations.py",
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PhotoEvaluationTests(unittest.TestCase):
    def test_report_contains_exact_model_image_and_timings_without_executing_content(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (300, 500), "brown").save(path)
            calls = []

            def analyze(image):
                calls.append(image)
                with Image.open(io.BytesIO(image)) as actual:
                    self.assertEqual(actual.format, "JPEG")
                return {"description": "<script>unsafe</script>", "observation": None}

            output = Path(directory) / "report.html"
            rows = evaluation.evaluate(
                [path], output, analyze, {"scope": "fixture_not_a_model_accuracy_test"}
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(rows[0]["status"], "unstructured")
            self.assertGreaterEqual(rows[0]["vision_call_s"], 0)
            self.assertGreaterEqual(rows[0]["total_s"], rows[0]["prepare_s"])
            report = output.read_text()
            self.assertIn("data:image/jpeg;base64,", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertNotIn("<script>", report)
            self.assertIn("Content-Security-Policy", report)
            with self.assertRaises(FileExistsError):
                evaluation.evaluate([path], output, analyze, {})

    def test_model_failure_records_type_but_not_secret_error_text_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.jpg" for i in range(2)]
            for path in paths:
                Image.new("RGB", (200, 400), "brown").save(path)

            def fail(image):
                raise RuntimeError("PRIVATE proxy PASSWORD")

            output = Path(directory) / "report.html"
            rows = evaluation.evaluate(paths, output, fail, {})
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["error_type"] == "RuntimeError" for row in rows))
            self.assertNotIn("PRIVATE", output.read_text())

    def test_bad_image_never_reaches_model_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jpg"
            path.write_bytes(b"not an image")

            def unexpected(image):
                self.fail("Bad image reached the model")

            row = evaluation.run_case(path, unexpected)
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error_type"], "ValueError")
            self.assertNotIn("image_base64", row)
