# Phân tích: Code hiện tại đã đủ để finetune SD3 bằng data SD302a chưa?

> [!NOTE]
> **Cập nhật 2026-08-18.** Cả 3 vấn đề critical đã được xử lý. Pipeline chạy thông từ
> `archive/` đến checkpoint MM-DiT. Hướng dẫn chạy đầy đủ: RUNBOOK.md (không có trong
> checkout hiện tại — xem cập nhật 2026-08-22 bên dưới).

> [!NOTE]
> **Cập nhật 2026-08-22 — máy mới (`aiserver`), repo là bản "upload" đã squash.**
> Repo hiện tại chỉ có 2 commit (`Create README.md`, `upload`) — mọi lịch sử/artefact cục
> bộ của lần phân tích trước đã bị gộp mất, kể cả `RUNBOOK.md`. Xem chi tiết ở cuối file.

## Kết luận

> [!TIP]
> **Đã đủ để chạy.** Dataset loader đọc đúng SD302a thật, CoarseNet.h5 đã convert và
> verify, Stage 1 caching hoạt động. Phần còn thiếu duy nhất là **compute**: Stage 1
> phải chạy trên GPU, và AFRNet cho `L_Identity` vẫn chưa có pretrained weights.

---

## ✅ Đã xử lý

### 1. ✅ Dataset loader khớp cấu trúc thật của SD302a

`{SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.png` trong `challengers/{DEVICE}/roll/png/`.

| | Trước | Sau |
|---|---|---|
| Filename regex | `(S\d+)_([Ff]\d+)_S([A-Ha-h])` | `(\d+)_([A-H])_([a-z]+)_(\d{1,2})` |
| Cấu trúc thư mục | flat | duyệt đệ quy `{DEVICE}/{IMPRESSION}/png/` |
| `sd302a_root` | `./data/NIST_SD302a` | `./archive/images/challengers` |
| Số sample load được | **0** (fallback 100 dummy) | **80.522 cặp** từ 13.630 ảnh |

Kết quả `python -m src.data.dataset`:

```
train:  72246 pairs | 181 subjects | 12286 unique source images
val  :   8276 pairs |  19 subjects |  1344 unique source images
```

Thêm vào:
- **Split theo subject** (`dataset.subject_split`, băm md5 của subject id) — 181/19
  subject không giao nhau. `train_mmdit.py` assert điều này. Trước đây `random_split`
  chia theo sample nên cùng một người rơi vào cả train lẫn val.
- **`contact_only`** đã thực sự được dùng để lọc.
- **`directed_pairs`** (A→B và B→A) + `max_pairs_per_finger` để giới hạn nếu cần.
- Ảnh thiếu file giờ **raise** thay vì trả về nhiễu ngẫu nhiên.

### 2. ✅ CoarseNet — port lại kiến trúc + convert weights

`CoarseNet.h5` (81 MB) đã có. Nhưng bản port PyTorch cũ **không khớp** kiến trúc gốc
nên không load được weights:

| Lỗi | Sửa |
|---|---|
| Thêm thừa `blockN_proj` trước mỗi residual block | Bỏ — trong MinutiaeNet `conv2_1` vừa là projection vừa là shortcut |
| `conv_block2` lấy trước MaxPool (1/4) | Lấy sau MaxPool (1/8), đúng như bản gốc |
| Thiếu hoàn toàn nhánh Gabor enhancement + minutiae | Port đủ: `enh_img_real/imag` → phase → `mnt_{o,w,h,s}` |
| BatchNorm eps 1e-5 (PyTorch) | 1e-3 (mặc định Keras) |
| Convert weights bằng cách đoán tên layer | Mỗi module mang `keras_name`, map chính xác |

```
Tensors mapped     : 432
Unmapped (PyTorch) : 0
Missing (Keras)    : 0
Parameters         : 20,124,478
Round-trip max diff: 0.000e+00
```

Thêm 3 lỗi ngữ nghĩa phát hiện khi verify bằng hình ảnh:

- **`seg_out` cao ở NỀN chứ không phải foreground.** Deploy script gốc dùng
  `1 - round(seg_out)`. Extractor đã đảo sẵn.
- **Orientation phải giải mã từ đỉnh đã làm mượt**, không phải trung bình có trọng số
  trên 90 sigmoid thô (vector tổng hợp chỉ ~0.36 → ra nhiễu, field nằm ngang toàn bộ).
  Đúng: `select_max(conv(ori_out, gausslabel))`.
- **Lớp Gabor không bất biến theo scale.** Kernel đã được finetune (mean 8.08, bias tới
  10.2) nên phải đưa vào ảnh 0–255 như MinutiaeNet, không phải [0,1].

Verify bằng `scripts/verify_coarsenet.py` (ghi ảnh ra `outputs/coarsenet_check/`):
orientation bám đúng luồng vân, 94 minutiae trên sensor A, 150 trên sensor F.

### 3. ✅ Stage 1 offline preprocessing

- Cache **theo ảnh** chứ không theo cặp: 13.630 file thay vì ~80.000.
- **Geometry contract**: ảnh vào canvas vuông `max(H,W)` ở tỉ lệ gốc, rồi resize 256.
  Stage 1 dùng đúng phép biến đổi đó → `S_aligned` khớp pixel với `img_A`.
- **Không resize trước khi chạy CoarseNet** — Gabor bank chỉnh cho ~500 ppi. Đo thực
  tế: sensor B–G ≈ 500 ppi, sensor H ≈ 930 ppi. `max_canvas: 768` chặn bộ nhớ và kéo
  H về gần 500 ppi.
- Idempotent, có `--limit` / `--overwrite` / `--no_minutiae`, in throughput + ETA.

### 4. ✅ Các mục phụ

| # | Việc | Trạng thái |
|---|---|---|
| 4 | `__init__.py` cho các module trong `src/` | ✅ đã thêm |
| 5 | `num_workers=0` hardcode | ✅ `dataset.num_workers` (mặc định 4) |
| 6 | Split không deterministic theo subject | ✅ băm subject id + assert chống rò rỉ |
| 7 | `contact_only` chưa implement | ✅ đã dùng |
| — | Checkpoint không lưu kiến trúc → `evaluate.py` vỡ sau `--small` | ✅ lưu `model_kwargs` |
| — | `evaluate.py` dùng CoarseNet random-init | ✅ dùng `build_extractor` |
| — | `train_vae.py` train trên 72k cặp (lặp ảnh) | ✅ train trên 12.286 ảnh duy nhất |

---

## ❌ Còn lại

### 1. 🔴 Compute — Stage 1 cần GPU

CoarseNet chạy ở độ phân giải gốc. Đo trên CPU (8 threads):

| Kích thước | Full (có minutiae) | Chỉ trunk |
|---|---|---|
| 512 px | 20 s | 3,9 s |
| 744 px | 166 s | 135 s |

→ 13.630 ảnh mất **hàng trăm giờ** trên CPU. Trên GPU khoảng **1–2 giờ**.

Máy hiện tại có GTX 1650 4 GB nhưng torch đang là bản CPU-only (`2.11.0+cpu`).

### 2. 🟠 AFRNet vẫn thiếu pretrained weights

`L_Identity` đang dùng embedder khởi tạo ngẫu nhiên → không bảo toàn identity.
Cần pretrain hoặc lấy một fingerprint embedding network có sẵn.

### 3. 🟡 TPS alignment luôn trả `is_aligned=False`

`ThinPlateSplineAligner` cần cặp điểm minutiae tương ứng giữa 2 sensor, hiện chưa có
bước matching nào sinh ra chúng. `S_aligned` vì vậy là structural map chưa đăng ký.
Extractor đã xuất được minutiae có hướng (`extract_minutiae`), nên bước còn lại là
matching giữa sensor A và B.

---

## 📋 Việc tiếp theo

| # | Task | Priority |
|---|---|---|
| 1 | Chạy Stage 1 đầy đủ trên GPU server (13.630 ảnh) | 🔴 Critical |
| 2 | Bật `dataset.require_cache: true` sau khi cache xong | 🔴 Critical |
| 3 | Train VAE (Phase 1) rồi MM-DiT (Phase 2) trên GPU | 🔴 Critical |
| 4 | Tìm/pretrain AFRNet cho `L_Identity` | 🟠 High |
| 5 | Minutiae matching giữa 2 sensor để TPS thực sự chạy | 🟡 Medium |

## Đã kiểm chứng (CPU, subset nhỏ)

| Bước | Kết quả |
|---|---|
| `verify_sd302a` | 13.630 file / 200 subject / 2.000 ngón |
| `convert_minutiaenet_weights` | 432/432 tensor, round-trip diff 0 |
| `verify_coarsenet` | orientation bám vân, minutiae hợp lý |
| `run_offline_preprocessing --limit 8` | `S_aligned` (6, 256, 256), seg ~0.3–0.4, ‖ori‖=1 |
| `train_vae` | ghi `weights/vae_fingerprint.pt` |
| `train_mmdit --small` | ghi `outputs/training/best.pt`, không rò rỉ subject |
| `evaluate` | ghi `outputs/eval/eval_results.json` |
| `pytest tests/` | 10 passed |

Số liệu chất lượng chưa có ý nghĩa — model mới train vài step trên CPU.

---

## 🔄 Cập nhật 2026-08-22 — máy `aiserver`, dataset đầy đủ tại đường dẫn mới

### Bối cảnh

Repo hiện tại (`git log`) chỉ có 2 commit: `Create README.md` rồi `upload` — một commit
duy nhất chứa toàn bộ 94 file, kể cả `__pycache__` và `outputs/`. Đây là bản **squash/export**
từ máy làm việc trước, nên mọi thứ *không track bằng git* (do `.gitignore` chặn `*.pt`, `*.pth`,
`outputs/` — dù `outputs/` vẫn lọt vào vì đã add trước khi ignore) đã **không** đi theo:

| Thiếu trên máy này | Được nhắc tới trong phân tích 08-18 |
|---|---|
| `RUNBOOK.md` | "Hướng dẫn chạy đầy đủ" |
| `archive/` (symlink/thư mục dataset cũ) | `sd302a_root` mặc định trỏ vào đây |
| `weights/vae_fingerprint.pt`, `weights/coarsenet_pytorch.pt` | output của `train_vae.py`, `convert_minutiaenet_weights.py` |
| `CoarseNet.h5` gốc (Keras) | input của bước convert |

`data/cached_stage1/` chỉ còn **8 file** (1 subject `00002303`, sensor A, 8 impression) — Stage 1
thực tế coi như chưa chạy trên máy này. `outputs/training/train_log.csv` (2 step) và
`outputs/eval/eval_results.json` (2 sample, minutiae precision/recall = 0, orientation RMSE
44°) vẫn là smoke-test cũ từ CPU, **không phản ánh chất lượng model**.

### ✅ Dataset tại `/home/aiserver/works/fingerprint/dataset/nist302a` — khớp hoàn toàn

Cấu trúc trên đĩa đúng layout SD302a chuẩn (`images/challengers/{A..H}/roll/png/...`), 2.0 GB:

| Sensor | A | B | C | D | E | F | G | H | **Tổng** |
|---|---|---|---|---|---|---|---|---|---|
| PNG | 1960 | 1956 | 1998 | 1987 | 1887 | 2000 | 700 | 1142 | **13.630** |

Trỏ thẳng `dataset.sd302a_root` vào đường dẫn này (không cần `archive/`) và chạy lại
`src/data/dataset.py` cho kết quả **giống hệt** báo cáo 08-18:

```
train:  72246 pairs | 181 subjects | 12286 unique source images
val  :   8276 pairs |  19 subjects |  1344 unique source images
```

→ `SD302aInspector`/`CrossSensorFingerprintDataset` hoạt động đúng với bản dataset này, không
cần sửa code — chỉ cần cập nhật `configs/default_config.yaml: dataset.sd302a_root`.

### 🖥️ Compute — vấn đề "🔴 Critical #1" của bản 08-18 giờ đã có lời giải

Máy này có **2× NVIDIA RTX A4000 16 GB** (driver 550.54, CUDA 12.4) và
`torch==2.1.0+cu121` với `torch.cuda.is_available() == True` — khác hẳn máy cũ
(GTX 1650 4 GB, torch CPU-only). Ước tính cũ "1–2 giờ trên GPU" cho 13.630 ảnh giờ khả thi.

### 🔎 Phát hiện thêm: có sẵn artefact thay thế trên cùng máy (ngoài repo)

Tìm trên filesystem thấy các file có khả năng vá 2 lỗ hổng còn lại, **nhưng nằm ngoài
project này** (thư mục làm việc khác của `binhan`/`pad`) — cần xin phép trước khi copy/dùng:

| File | Kích thước | Ghi chú |
|---|---|---|
| `~/workspace/binhan/fingerprint/recognition/MinutiaeNet/Models/CoarseNet.h5` | 81.112.872 bytes | **Khớp chính xác** với "81 MB, round-trip diff 0" đã verify trong báo cáo 08-18 → gần như chắc chắn là cùng 1 file gốc |
| `~/workspace/binhan/fingerprint/recognition/AFR-Net/ckpt/best.pt` / `best.pth` | 558 MB / 718 MB | Checkpoint AFR-Net đã train (dict `{model, optim}`) — ứng viên cho `L_Identity`, nhưng **chưa kiểm tra khớp kiến trúc** với `AFRNetEmbeddingExtractor` trong `src/losses/identity_loss.py` |
| `~/workspace/pad/finger_gen/model_bench/weights/afrnet.pth` | 190 MB | Một checkpoint AFRNet khác, cũng cần kiểm tra tương thích |

Việc "🟠 High #2 — AFRNet thiếu pretrained weights" ở báo cáo 08-18 do đó **có thể không còn
là vấn đề thiếu weights, mà là vấn đề adapt kiến trúc** — cần đối chiếu `AFRNetEmbeddingExtractor`
với `afrnet/model.py` trong repo AFR-Net để viết converter, tương tự cách đã làm với CoarseNet.

### 📋 Việc tiếp theo (cập nhật độ ưu tiên)

| # | Task | Trạng thái |
|---|---|---|
| 1 | Sửa `configs/default_config.yaml: dataset.sd302a_root` → `/home/aiserver/works/fingerprint/dataset/nist302a/images/challengers` | 🔴 Chưa làm, 1 dòng config |
| 2 | Lấy lại `CoarseNet.h5` (copy từ `MinutiaeNet/Models/` nếu được phép) + chạy `scripts/convert_minutiaenet_weights.py` để có `weights/coarsenet_pytorch.pt` | 🔴 Blocker cho Stage 1 |
| 3 | Chạy `scripts/run_offline_preprocessing.py` full 13.630 ảnh trên GPU A4000 | 🔴 Critical, giờ khả thi (~1–2h) |
| 4 | Bật `dataset.require_cache: true` sau khi cache xong | 🔴 Critical |
| 5 | Train VAE rồi MM-DiT thật trên GPU (không phải smoke test) | 🔴 Critical |
| 6 | Đối chiếu `AFR-Net/ckpt/best.pt` với `AFRNetEmbeddingExtractor`, viết converter nếu khớp | 🟠 High |
| 7 | Minutiae matching giữa 2 sensor để TPS `is_aligned` thực sự `True` | 🟡 Medium |
| 8 | Tái tạo/khôi phục `RUNBOOK.md` (bị mất trong lần upload này) | 🟢 Low |
