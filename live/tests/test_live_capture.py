"""Smoke tests for REAL JPEG intake, file saving, and capture-only mode.

These tests use a generated JPEG fixture; they do NOT simulate a physical
OV2640 sensor and they do NOT test the PalmBridge checkpoint's accuracy.

Run: python -m unittest discover -s live/tests -v
"""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import live.live_web as live


class CameraReply(io.BytesIO):
    status = 200


class TestLiveCapture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir = live.CAPTURES
        live.CAPTURES = Path(self.tmp.name)
        self.app = live.build_app(SimpleNamespace(
            capture_only=True, camera_url="http://192.168.0.123/capture"
        ))
        self.client = self.app.test_client()
        image = Image.new("RGB", (80, 80), color=(205, 180, 150))
        buff = io.BytesIO()
        image.save(buff, format="JPEG")
        self.jpeg = buff.getvalue()

    def tearDown(self):
        live.CAPTURES = self.old_dir
        self.tmp.cleanup()

    def test_mock_camera_jpeg_saved(self):
        with patch("live.live_web.urllib.request.urlopen",
                   return_value=CameraReply(self.jpeg)):
            response = self.client.post("/api/capture?expected=person_01")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        data = response.json
        self.assertTrue(data["image_url"].endswith("_esp32cam.jpg"))
        self.assertIsNone(data["predicted"])
        files = list(Path(self.tmp.name).glob("*.jpg"))
        self.assertEqual(len(files), 1)
        with Image.open(files[0]) as photo:
            self.assertEqual(photo.size, (80, 80))

    def test_browser_upload_saved(self):
        response = self.client.post("/api/upload", data={
            "file": (io.BytesIO(self.jpeg), "webcam.jpg"),
            "expected": "person_04"
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.jpg"))), 1)

    def test_invalid_upload_rejected(self):
        response = self.client.post("/api/upload", data={
            "file": (io.BytesIO(b"not a photo"), "bad.jpg")
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(Path(self.tmp.name).glob("*.jpg")), [])


if __name__ == "__main__":
    unittest.main()
