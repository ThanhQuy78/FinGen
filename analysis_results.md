# Phân tích: Code hiện tại đã đủ để finetune SD3 bằng data SD302a chưa?

> [!NOTE]
> **Cập nhật 2026-08-18.** Cả 3 vấn đề critical đã được xử lý. Pipeline chạy thông từ
> `archive/` đến checkpoint MM-DiT. Hướng dẫn chạy đầy đủ: [RUNBOOK.md](RUNBOOK.md).

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
