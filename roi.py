"""Detect a hand and extract an oriented palm ROI with MediaPipe landmarks."""
from pathlib import Path
import ctypes
import math
import os
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageOps


def _windows_ascii_path(path):
    """Return an ASCII 8.3 path for native MediaPipe on Windows."""
    value = str(Path(path).resolve())
    if os.name != "nt" or value.isascii():
        return value
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(
        value, buffer, len(buffer)
    )
    short_value = buffer.value
    if length == 0 or not short_value or not short_value.isascii():
        raise RuntimeError(
            "MediaPipe trên Windows cần đường dẫn ASCII. Windows chưa tạo "
            "được đường dẫn ngắn 8.3 cho thư mục tạm."
        )
    return short_value


class HandPalmROICropper:
    """Crop the palm between the wrist and the finger knuckles.

    The crop is aligned using the wrist and MCP knuckles, then mapped to a
    square image. The scale values add margin around the palm and can be
    adjusted after inspecting the debug frame produced by ``crop-roi``.
    """

    def __init__(self, model_path, width_scale=1.30, height_scale=1.50,
                 output_size=256):
        self.model_path = Path(model_path)
        self.width_scale = float(width_scale)
        self.height_scale = float(height_scale)
        self.output_size = int(output_size)
        if self.width_scale <= 0 or self.height_scale <= 0:
            raise ValueError("ROI width/height scale phải lớn hơn 0.")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Không thấy model MediaPipe: {self.model_path}\n"
                "Tải hand_landmarker.task theo hướng dẫn trong README.md."
            )
        self._runtime_dir = None
        self._landmarker = None
        native_model_path = str(self.model_path.resolve())
        native_working_dir = None
        try:
            if os.name == "nt":
                self._runtime_dir = Path(tempfile.mkdtemp(prefix="palm_roi_"))
                runtime_model = self._runtime_dir / "hand_landmarker.task"
                shutil.copyfile(self.model_path, runtime_model)
                native_model_path = _windows_ascii_path(runtime_model)
                native_working_dir = _windows_ascii_path(self._runtime_dir)

            old_working_dir = os.getcwd()
            try:
                if native_working_dir is not None:
                    os.chdir(native_working_dir)
                import mediapipe as mp

                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=native_model_path
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                )
                self._landmarker = (
                    mp.tasks.vision.HandLandmarker.create_from_options(options)
                )
            finally:
                if native_working_dir is not None:
                    os.chdir(old_working_dir)
            self._mp = mp
        except ImportError as exc:
            self._cleanup_runtime_dir()
            raise RuntimeError(
                "Thiếu MediaPipe. Chạy: pip install -r requirements.txt"
            ) from exc
        except Exception:
            self._cleanup_runtime_dir()
            raise

    @staticmethod
    def _unit(vector):
        length = math.hypot(vector[0], vector[1])
        if length < 1e-6:
            raise ValueError("Không xác định được hướng lòng bàn tay.")
        return vector[0] / length, vector[1] / length

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def crop(self, image):
        """Return ``(palm_roi, frame_quad)`` from a PIL image.

        ``frame_quad`` is ordered top-left, bottom-left, bottom-right,
        top-right in the original image, with the fingers at the top of ROI.
        """
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        rgb = np.ascontiguousarray(np.asarray(image))
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            raise ValueError(
                "Không phát hiện được bàn tay. Hãy chụp rõ một bàn tay, "
                "đủ cổ tay và các khớp gốc ngón tay."
            )

        candidates = []
        for landmarks in result.hand_landmarks:
            points = [(lm.x * width, lm.y * height) for lm in landmarks]
            wrist, index_mcp, middle_mcp, pinky_mcp = (
                points[0], points[5], points[9], points[17]
            )
            palm_width = self._distance(index_mcp, pinky_mcp)
            palm_length = self._distance(wrist, middle_mcp)
            if palm_width >= 8 and palm_length >= 8:
                candidates.append((palm_width * palm_length, points))
        if not candidates:
            raise ValueError("Các điểm bàn tay phát hiện được không đủ rõ để crop ROI.")

        _, points = max(candidates, key=lambda item: item[0])
        wrist, index_mcp, middle_mcp, pinky_mcp = (
            points[0], points[5], points[9], points[17]
        )
        palm_center = (
            (wrist[0] + middle_mcp[0]) / 2,
            (wrist[1] + middle_mcp[1]) / 2,
        )
        axis_y = self._unit((middle_mcp[0] - wrist[0],
                             middle_mcp[1] - wrist[1]))
        axis_x = (-axis_y[1], axis_y[0])
        index_to_pinky = (pinky_mcp[0] - index_mcp[0],
                          pinky_mcp[1] - index_mcp[1])
        if axis_x[0] * index_to_pinky[0] + axis_x[1] * index_to_pinky[1] < 0:
            axis_x = (-axis_x[0], -axis_x[1])

        crop_width = self._distance(index_mcp, pinky_mcp) * self.width_scale
        crop_height = self._distance(wrist, middle_mcp) * self.height_scale
        half_x = (axis_x[0] * crop_width / 2, axis_x[1] * crop_width / 2)
        half_y = (axis_y[0] * crop_height / 2, axis_y[1] * crop_height / 2)
        cx, cy = palm_center
        frame_quad = (
            (cx - half_x[0] + half_y[0], cy - half_x[1] + half_y[1]),
            (cx - half_x[0] - half_y[0], cy - half_x[1] - half_y[1]),
            (cx + half_x[0] - half_y[0], cy + half_x[1] - half_y[1]),
            (cx + half_x[0] + half_y[0], cy + half_x[1] + half_y[1]),
        )
        quad_data = tuple(value for point in frame_quad for value in point)
        roi = image.transform(
            (self.output_size, self.output_size),
            Image.Transform.QUAD,
            quad_data,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )
        return roi, frame_quad

    def crop_file(self, image_path):
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        roi, frame_quad = self.crop(image)
        return image, roi, frame_quad

    def close(self):
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        finally:
            self._cleanup_runtime_dir()

    def _cleanup_runtime_dir(self):
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
            self._runtime_dir = None
