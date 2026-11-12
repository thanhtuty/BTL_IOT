"""Local Streamlit UI: upload ESP32-CAM images of four people and fine-tune.

Run from repo root: streamlit run train_app.py
Images, ROI, checkpoints remain on the user's computer (Git ignored).
"""
import io
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from config import DEFAULT_HAND_LANDMARKER, DEFAULT_WEIGHTS, ROOT, SUPPORTED_EXTENSIONS
from finetune import finetune, prepare_data

RAW = ROOT / 'dataset' / 'esp32_uploads'
PREPARED = ROOT / 'dataset' / 'esp32_prepared'
OUTPUT = ROOT / 'weights' / 'palmbridge_esp32_finetuned.pt'
HAND_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_BYTES = 250 * 1024 * 1024

st.set_page_config(page_title='PalmLock • Train 4 người', page_icon='✋', layout='wide')
st.title('✋ PalmLock — Fine-tune từ ảnh ESP32-CAM')
st.caption('Xử lý và train hoàn toàn trên máy này; không tải ảnh lên GitHub hay máy chủ ngoài.')


def safe_name(value):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', value)[:65] or 'image'


def write_image(content, target):
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f'Ảnh vượt 12 MB: {target.name}')
    try:
        with Image.open(io.BytesIO(content)) as original:
            original.verify()
        with Image.open(io.BytesIO(content)) as original:
            img = ImageOps.exif_transpose(original).convert('RGB')
            if img.width * img.height > 16_000_000:
                raise ValueError('Ảnh quá lớn (>16 triệu pixels).')
            target.parent.mkdir(parents=True, exist_ok=True)
            img.save(target, format='JPEG', quality=95)
    except Exception as exc:
        raise ValueError(f'{target.name}: không đọc được ảnh ({exc})') from exc


def store_files(upload_map):
    # Reserved folder created ONLY by this UI. Existing project dataset/ is untouched.
    if RAW.exists():
        shutil.rmtree(RAW)
    totals = {}
    all_bytes = 0
    for person, files in upload_map.items():
        count, sessions = 0, 0
        loose = []
        for n, uploaded in enumerate(files):
            contents = uploaded.getvalue()
            all_bytes += len(contents)
            if all_bytes > MAX_TOTAL_BYTES:
                raise ValueError('Tổng dữ liệu vượt 250 MB; chia thành nhiều lần chụp.')
            if uploaded.name.lower().endswith('.zip'):
                sessions += 1
                session = RAW / person / f'session_{sessions:02d}_zip_{n:02d}'
                with zipfile.ZipFile(io.BytesIO(contents)) as archive:
                    entries = [z for z in archive.infolist()
                               if not z.is_dir()
                               and Path(z.filename).suffix.lower() in SUPPORTED_EXTENSIONS]
                    if len(entries) > 300:
                        raise ValueError('Mỗi ZIP hỗ trợ tối đa 300 ảnh.')
                    for k, entry in enumerate(entries):
                        if entry.file_size > MAX_IMAGE_BYTES:
                            raise ValueError(f'Ảnh quá lớn trong ZIP: {entry.filename}')
                        data = archive.read(entry)
                        name = safe_name(Path(entry.filename).stem)
                        write_image(data, session / f'{k:04d}_{name}.jpg')
                        count += 1
            elif Path(uploaded.name).suffix.lower() in SUPPORTED_EXTENSIONS:
                loose.append((n, uploaded.name, contents))
        if loose:
            sessions += 1
            session = RAW / person / f'session_{sessions:02d}_loose'
            for n, name, data in loose:
                write_image(data, session / f'{n:04d}_{safe_name(Path(name).stem)}.jpg')
                count += 1
        totals[person] = {'photos': count, 'sessions': sessions}
    return totals


with st.expander('1. Hướng dẫn chuẩn bị ảnh', expanded=True):
    st.markdown('''
    **Chỉ cần chọn ảnh của 4 người phía dưới.** Có thể chọn JPG/PNG hoặc ZIP.

    - Khuyến nghị **ít nhất 3 ZIP chụp ở 3 buổi/lượt khác nhau cho mỗi người**.
      Mỗi ZIP được tính là một lượt chụp, giúp chia train/validation/test độc lập.
    - Ảnh **toàn bàn tay**: chọn chế độ *Tự cắt ROI (MediaPipe)*, cần nhìn rõ cổ tay và các khớp ngón.
    - Ảnh **đã cắt lòng bàn tay**: chọn chế độ *ROI đã cắt*.
    - Ảnh chụp liên tiếp trong một ZIP **không nên phân tán vào các tập train và test**.
    ''')

uploads = {}
cols = st.columns(2)
for i in range(4):
    with cols[i % 2]:
        st.subheader(f'Người {i + 1}')
        uploads[f'person_{i + 1:02d}'] = st.file_uploader(
            f'Chọn ảnh/ZIP người {i + 1}', type=['jpg', 'jpeg', 'png', 'webp', 'zip'],
            accept_multiple_files=True, key=f'person_{i+1}'
        )

st.divider()
st.subheader('2. Cấu hình huấn luyện')
col1, col2, col3 = st.columns(3)
with col1:
    mode_label = st.radio('Ảnh đầu vào',
                          ['Toàn bàn tay (tự cắt ROI)', 'ROI đã cắt sẵn'],
                          horizontal=False)
    mode = 'full-hand' if mode_label.startswith('Toàn') else 'roi'
with col2:
    epochs = st.slider('Số epoch', min_value=1, max_value=50, value=12)
    batch_size = st.selectbox('Batch size', [2, 4, 8, 16], index=1)
with col3:
    allow_random = st.checkbox('Cho phép chạy thử khi <3 lượt chụp/người', value=False,
                               help='Chỉ dùng để thử kỹ thuật. Kết quả test có thể bị thổi phồng do ảnh liên tiếp.')

if not DEFAULT_WEIGHTS.exists():
    st.error('Thiếu checkpoint weights/best_palmbridge_V1_embedding.pt. '
             'Hãy đặt trọng số PalmBridge gốc vào thư mục weights/ trước khi train.')

if mode == 'full-hand' and not DEFAULT_HAND_LANDMARKER.exists():
    st.info('Lần đầu xử lý ảnh toàn bàn tay cần tải hand_landmarker.task của Google. '
            'Ứng dụng sẽ tự tải **model** (không gửi ảnh đi) khi bấm Train, nếu máy có Internet.')

if st.button('🚀 Lưu ảnh và bắt đầu Fine-tune', type='primary', use_container_width=True):
    if not all(uploads.values()):
        st.error('Bạn cần chọn ảnh cho đủ cả 4 người.')
    elif not DEFAULT_WEIGHTS.is_file():
        st.error('Chưa có checkpoint PalmBridge gốc (.pt).')
    else:
        try:
            with st.status('Đang xử lý dữ liệu và huấn luyện...', expanded=True) as status:
                st.write('Bước 1: Đọc ảnh và các file ZIP')
                totals = store_files(uploads)
                st.json(totals)
                if mode == 'full-hand' and not DEFAULT_HAND_LANDMARKER.exists():
                    st.write('Đang tải model MediaPipe (chỉ tải file mô hình)...')
                    DEFAULT_HAND_LANDMARKER.parent.mkdir(exist_ok=True, parents=True)
                    urllib.request.urlretrieve(HAND_MODEL_URL, DEFAULT_HAND_LANDMARKER)
                st.write('Bước 2: Cắt ROI và chia train/val/test')
                report = prepare_data(RAW, PREPARED, input_format=mode,
                                      allow_random_split=allow_random)
                st.json(report['counts'])
                if report['demo_only']:
                    st.warning('Đây là phép thử với tách ảnh ngẫu nhiên; '
                               'validation/test không đáng tin cho triển khai thực tế.')
                if report['skipped']:
                    st.warning(f"Đã bỏ qua {len(report['skipped'])} ảnh không hợp lệ/crop lỗi.")
                    with st.expander('Xem ảnh bị bỏ qua'):
                        st.text('\n'.join(report['skipped'][:100]))
                st.write('Bước 3: Fine-tune PalmBridge')
                bar = st.progress(0.)
                epoch_text = st.empty()
                def on_epoch(row, epoch, total):
                    bar.progress(epoch / total)
                    epoch_text.text(f"Epoch {epoch}/{total} | train={row['train_acc']:.1%} "
                                    f"val={row['val_acc']:.1%} | val_loss={row['val_loss']:.4f}")
                result = finetune(PREPARED, weights=DEFAULT_WEIGHTS, output=OUTPUT,
                                  epochs=epochs, batch_size=batch_size,
                                  progress=on_epoch)
                status.update(label='Fine-tune hoàn tất', state='complete')
            st.success('Đã lưu trọng số mới: weights/palmbridge_esp32_finetuned.pt')
            st.metric('Validation accuracy tốt nhất', f"{result['best_val_acc']:.1%}")
            st.metric('Test accuracy (4 người đã biết)', f"{result['test_closed_set_acc']:.1%}")
            st.warning('Accuracy trên 4 người KHÔNG đồng nghĩa từ chối được người lạ. '
                       'Phải kiểm thử người lạ và chọn ngưỡng trước khi điều khiển khóa.')
        except Exception as exc:
            st.error(f'Train thất bại: {exc}')
            st.exception(exc)

st.caption('Dữ liệu nằm trong dataset/esp32_uploads và dataset/esp32_prepared; '
           'checkpoint ở weights/. Các thư mục này không được commit lên GitHub.')

if OUTPUT.is_file():
    st.divider()
    st.download_button('📥 Tải lại checkpoint đã huấn luyện (.pt)',
                       OUTPUT.read_bytes(), file_name='palmbridge_esp32_finetuned.pt')