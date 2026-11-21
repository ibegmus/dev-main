import asyncio
import base64
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

import api
import db
import files
from request_queue import InferenceQueue, QueueFull


def photo_bytes(format="PNG", mode="RGB", size=(24, 16), **kwargs):
    output = io.BytesIO()
    Image.new(mode, size).save(output, format=format, **kwargs)
    return output.getvalue()


class PhotoTests(unittest.TestCase):
    def test_supported_formats_normalize_to_jpeg(self):
        for format in ["JPEG", "PNG", "WEBP"]:
            with self.subTest(format=format):
                encoded = base64.b64encode(photo_bytes(format)).decode()
                with Image.open(io.BytesIO(files.decode_photo(encoded))) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")

    def test_camera_rotation_and_transparency(self):
        exif = Image.Exif()
        exif[274] = 6
        data = photo_bytes("JPEG", exif=exif)
        decoded = files.decode_photo(base64.b64encode(data).decode())
        with Image.open(io.BytesIO(decoded)) as image:
            self.assertEqual(image.size, (16, 24))
        transparent = photo_bytes("PNG", "RGBA")
        decoded = files.decode_photo(base64.b64encode(transparent).decode())
        with Image.open(io.BytesIO(decoded)) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_bad_payloads_do_not_reach_model(self):
        for encoded in ["%%%", base64.b64encode(b"not an image").decode(),
                        base64.b64encode(photo_bytes("GIF")).decode()]:
            with self.subTest(encoded=encoded[:20]), self.assertRaises(ValueError):
                files.decode_photo(encoded)

    def test_image_size_limits(self):
        with self.assertRaises(OverflowError):
            files.decode_photo(base64.b64encode(b"x" * (files.MAX_PHOTO_SIZE + 1)).decode())
        large = photo_bytes("PNG", mode="L", size=(5000, 4001))
        with self.assertRaisesRegex(ValueError, "мегапиксел"):
            files.decode_photo(base64.b64encode(large).decode())
        resized = files.decode_photo(base64.b64encode(photo_bytes(size=(2400, 1200))).decode())
        with Image.open(io.BytesIO(resized)) as image:
            self.assertEqual(image.size, (1600, 800))

    def test_vision_model_receives_the_actual_image(self):
        jpeg = photo_bytes("JPEG")
        model = SimpleNamespace(invoke=Mock(return_value=SimpleNamespace(content="Слои почвы")))
        with patch.object(files, "vision_llm", model):
            self.assertEqual(files.describe_photo(jpeg), "Слои почвы")
        message = model.invoke.call_args.args[0][0]
        image_url = message.content[1]["image_url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), jpeg)

    def test_photo_description_is_used_by_the_answering_agent(self):
        graph = SimpleNamespace(invoke=Mock(return_value={"answer": "Ответ по фото"}))
        vision = SimpleNamespace(invoke=Mock(return_value=SimpleNamespace(content="Тёмный верхний слой")))
        with patch.dict(sys.modules, {"agent": SimpleNamespace(app=graph)}), patch.object(files, "vision_llm", vision):
            answer = api.run_agent("Какие слои?", photo_bytes("JPEG"))
        self.assertEqual(answer["answer"], "Ответ по фото")
        self.assertEqual(graph.invoke.call_args.args[0], {
            "question": "Какие слои?", "image_description": "Тёмный верхний слой"
        })


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_positions_and_single_inference(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def process(job):
            calls.append(job.text)
            if job.text == "first":
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release inference")
            return job.text

        queue = InferenceQueue(process)
        await queue.start()
        try:
            first = queue.submit("first", b"photo", None)
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            second = queue.submit("second", None, None)
            third = queue.submit("third", None, None)
            self.assertEqual(queue.snapshot(first.id)["status"], "processing")
            self.assertEqual(queue.snapshot(second.id)["ahead"], 1)
            self.assertEqual(queue.snapshot(third.id)["ahead"], 2)
            self.assertEqual(calls, ["first"])
            # Disconnecting a waiter must not release the inference slot.
            waiter = asyncio.create_task(first.done.wait())
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertEqual(queue.snapshot(third.id)["ahead"], 2)
            release.set()
            await asyncio.wait_for(third.done.wait(), 5)
            self.assertEqual(calls, ["first", "second", "third"])
            self.assertEqual(queue.snapshot(first.id)["answer"], "first")
            self.assertIsNone(first.image)
        finally:
            release.set()
            await queue.close()

    async def test_failure_does_not_stall_queue(self):
        def process(job):
            if job.text == "bad":
                raise RuntimeError("Model unavailable")
            return "answer"

        queue = InferenceQueue(process)
        await queue.start()
        try:
            bad = queue.submit("bad", b"photo", None)
            good = queue.submit("good", None, None)
            with self.assertLogs("request_queue", level="ERROR") as logs:
                await asyncio.wait_for(good.done.wait(), 5)
            self.assertNotIn(bad.id, '\n'.join(logs.output))
            self.assertIn(bad.id[:8], '\n'.join(logs.output))
            self.assertEqual(queue.snapshot(bad.id)["status"], "failed")
            self.assertEqual(queue.snapshot(good.id)["answer"], "answer")
            self.assertIsNone(bad.image)
        finally:
            await queue.close()

    async def test_capacity_and_result_expiry(self):
        queue = InferenceQueue(lambda job: "answer", max_pending=1, result_ttl=0)
        await queue.start()
        try:
            job = queue.submit("one", None, None)
            with self.assertRaises(QueueFull):
                queue.submit("two", None, None)
            await asyncio.wait_for(job.done.wait(), 5)
            with self.assertRaises(KeyError):
                queue.snapshot(job.id)
        finally:
            await queue.close()


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db, "DB_FILE", str(Path(self.temp.name) / "app.db"))
        self.db_patch.start()
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.received = []

        def run_agent(text, image):
            self.received.append((text, image))
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("Test did not release inference")
            return {"answer": "Ответ модели", "intent": "science"}

        self.model_patch = patch.object(api, "run_agent", side_effect=run_agent)
        self.model_patch.start()
        self.preload_patch = patch.object(api, "preload_reranker")
        self.preload_patch.start()
        self.client = TestClient(api.app)
        self.client.__enter__()

    def tearDown(self):
        self.release.set()
        self.client.__exit__(None, None, None)
        self.model_patch.stop()
        self.preload_patch.stop()
        self.db_patch.stop()
        self.temp.cleanup()

    def wait_for_result(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.client.get(f"/ask/jobs/{job_id}").json()
            if result["status"] in {"done", "failed"}:
                return result
            time.sleep(0.01)
        self.fail("Job did not finish")

    def test_photo_only_is_processed_and_saved_for_logged_in_user(self):
        self.client.post("/register", json={"name": "Test", "email": "test@example.invalid", "password": "Test-for-photo-12345", "terms_accepted": True})
        token = self.client.post("/login", json={"email": "test@example.invalid", "password": "Test-for-photo-12345"}, headers={"X-TerraMind-Request": "1"}).json()["token"]
        encoded = base64.b64encode(photo_bytes()).decode()
        response = self.client.post("/ask/jobs", json={"image": encoded}, headers={"x-token": token, "X-TerraMind-Request": "1"})
        self.assertEqual(response.status_code, 202)
        result = self.wait_for_result(response.json()["job_id"])
        self.assertEqual(result["answer"], "Ответ модели")
        question, image = self.received[0]
        self.assertIn("фото", question)
        with Image.open(io.BytesIO(image)) as normalized:
            self.assertEqual(normalized.format, "JPEG")
        history = self.client.get("/history", headers={"x-token": token, "X-TerraMind-Request": "1"}).json()["history"]
        self.assertEqual(history[0]["answer"], "Ответ модели")
        self.assertIn("[с фото]", history[0]["question"])

    def test_two_clients_get_real_positions_while_api_stays_responsive(self):
        self.release.clear()
        first = self.client.post("/ask/jobs", json={"text": "first"}).json()
        self.assertTrue(self.started.wait(5))
        self.client.cookies.clear()  # Independent guest visitor, same API loop.
        second = self.client.post("/ask/jobs", json={"text": "second"}).json()
        self.client.cookies.clear()
        third = self.client.post("/ask/jobs", json={"text": "third"}).json()
        self.assertEqual(self.client.get(f"/ask/jobs/{second['job_id']}").json()["ahead"], 1)
        self.assertEqual(self.client.get(f"/ask/jobs/{third['job_id']}").json()["ahead"], 2)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.release.set()
        self.assertEqual(self.wait_for_result(third["job_id"])["status"], "done")
        self.assertEqual([text for text, _ in self.received], ["first", "second", "third"])

    def test_invalid_requests_are_rejected_before_model(self):
        self.assertEqual(self.client.post("/ask/jobs", json={"text": "  "}).status_code, 400)
        self.assertEqual(self.client.post("/ask/jobs", json={"image": "%%%"}).status_code, 400)
        encoded = base64.b64encode(b"x" * (files.MAX_PHOTO_SIZE + 1)).decode()
        self.assertEqual(self.client.post("/ask/jobs", json={"image": encoded}).status_code, 413)
        self.assertEqual(self.client.get("/ask/jobs/not-a-real-id").status_code, 404)
        self.assertEqual(self.received, [])

    def test_queue_capacity_returns_retryable_error(self):
        self.release.clear()
        self.client.post("/ask/jobs", json={"text": "active"})
        self.assertTrue(self.started.wait(5))
        api.app.state.question_queue.max_pending = 1
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/ask/jobs", json={"text": "waiting"}).status_code, 202)
        self.client.cookies.clear()
        response = self.client.post("/ask/jobs", json={"text": "full"})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "10")

    def test_legacy_ask_keeps_answer_contract(self):
        response = self.client.post("/ask", json={"text": "Вопрос"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "Ответ модели"})


if __name__ == "__main__":
    unittest.main()
