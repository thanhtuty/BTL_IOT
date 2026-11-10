"""Tiền xử lý PalmBridge và chuẩn hóa ảnh đầu vào.

Mặc định ảnh phải là ROI đã crop. Có thể truyền HandPalmROICropper để lấy ROI
từ ảnh toàn bàn tay trước khi áp dụng transform.
"""
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from config import IMAGE_SIZE

class NormSingleROI:
    """
    Normalize non-black pixels to zero mean and unit variance.
    This follows the preprocessing logic in the supplied
    PalmBridge/COMPNet code.
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()

        if c != 1:
            raise TypeError(
                "NormSingleROI requires a grayscale 1-channel image."
            )

        tensor = tensor.view(c, h * w)

        idx = tensor > 0
        t = tensor[idx]

        if t.numel() > 0:
            mean_value = t.mean()
            std_value = t.std()

            t = (
                t.sub(mean_value)
                .div(std_value + 1e-6)
            )

            tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(
                tensor,
                repeats=self.outchannels,
                dim=0,
            )

        return tensor


def build_transform():
    # Đúng thứ tự transform trong palmbridge.ipynb cell 2.
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        NormSingleROI(),
    ])


def preprocess_file(path, transform=None, cropper=None):
    """Trả về tensor [1, 1, 128, 128] có thể đưa vào model."""
    transform = transform or build_transform()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if cropper is not None:
            image, _ = cropper.crop(image)
        roi = image.convert("L")
        tensor = transform(roi)
    if torch.count_nonzero(tensor).item() < 4:
        raise ValueError("ROI gần như rỗng/đen; kiểm tra ảnh đầu vào hoặc cắt ROI.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("ROI tạo ra NaN/Inf sau chuẩn hóa; kiểm tra ảnh.")
    return tensor.unsqueeze(0)
