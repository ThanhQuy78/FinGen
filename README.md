# FinGen — Cross-Sensor Fingerprint Generation & Transfer

Sinh ảnh vân tay từ sensor A sang sensor B (cùng ngón, khác thiết bị chụp) bằng
diffusion transformer 2 luồng (dual-stream MM-DiT), điều kiện hoá bởi bản đồ cấu
trúc vân tay (segmentation + orientation + minutiae) trích từ CoarseNet.

> Lịch sử debug/quyết định chi tiết: xem [analysis_results.md](analysis_results.md).
> File này là tài liệu tham chiếu trạng thái hiện tại + hướng dẫn chạy.

---

## 1. Kiến trúc tổng quan

```
NIST SD302a (13.630 ảnh, 8 sensor A-H)
        │
        ▼
┌───────────────────────┐
│ Stage 1: CoarseNet     │  segmentation + orientation + minutiae
│ (offline, cache 1 lần) │  → S_aligned (6, 256, 256) / ảnh
└───────────┬───────────┘
            │
┌───────────▼───────────┐        ┌─────────────────────────┐
│ Phase 1: VAE           │        │ Phase 2: MM-DiT          │
│ ảnh (1,256,256)         │──────▶│ Stream X: latent nhiễu    │
│  ↔ latent (4,32,32)     │ frozen│ Stream Y: S_aligned       │
└─────────────────────────┘        │ Rectified Flow            │
                                    │ L_Diff + L_Identity(DMD)  │
                                    │       + L_Orient          │
                                    └────────────┬─────────────┘
                                                 │
                                    ┌────────────▼─────────────┐
                                    │ Inference: Euler sampling │
                                    │ → VAE decode → ảnh sinh   │
                                    └───────────────────────────┘
```

**MM-DiT ở đây là kiến trúc tự thiết kế**, không phải finetune từ Stable
Diffusion 3 — không load pretrained weight nào của SD3. Chỉ mượn khái niệm
"dual-stream transformer + AdaLN modulation". Khác biệt quan trọng nhất: SD3 dùng
joint bidirectional attention giữa 2 stream, còn ở đây Stream X chỉ **cross-attention
1 chiều** sang Stream Y (`src/models/mm_dit.py`, class `MMDiTBlock`).

---

## 2. Dataset — NIST SD302a

Cấu trúc thật trên đĩa (`src/data/sd302a_inspector.py`):

```
{root}/{DEVICE}/{IMPRESSION}/png/{SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.png
vd:    challengers/A/roll/png/00002303_A_roll_01.png
```

- 8 sensor contact-based: A (optical), B (optical), C (capacitive), D (thermal-sweep),
  E (optical), F (capacitive), G (optical rolled), H (capacitive-sweep)
- 13.630 ảnh, 200 subject, ~2.000 ngón (10 FRGP/subject)
- Split **theo subject** (băm md5), không theo sample — tránh rò rỉ identity giữa
  train/val (`dataset.subject_split` trong `src/data/dataset.py`)
- Cặp cross-sensor: với mỗi ngón, mọi cặp sensor `(s_i, s_j)` xuất hiện 2 chiều
  (`directed_pairs: true`) → **80.515 cặp** (72.239 train / 8.276 val, sau khi lọc
  7 cặp tham chiếu 1 ảnh nguồn hỏng thật — file PNG 1×1 px, 88 byte, lỗi có sẵn
  trong bản gốc SD302a)

Đường dẫn dataset hiện dùng: `/home/aiserver/works/fingerprint/dataset/nist302a/images/challengers`
(cấu hình ở `configs/default_config.yaml: dataset.sd302a_root`).

---

## 3. Stage 1 — CoarseNet structural extraction (offline, cache 1 lần)

`src/preprocessing/fingernet_extractor.py::FingerNetExtractor` — port PyTorch của
CoarseNet (MinutiaeNet), weight convert từ `CoarseNet.h5` gốc (432/432 tensor khớp,
round-trip diff 0).

Output 3 bản đồ, gộp thành `S` (6 kênh):

| Kênh | Ý nghĩa |
|---|---|
| 1 | segmentation (foreground probability) |
| 2 (1:3) | orientation `(cos2θ, sin2θ)` |
| 3 (3:6) | minutiae `(score, cos φ, sin φ)` |

**Chạy ở độ phân giải gốc** (không resize trước) vì Gabor bank tuned cho ~500 ppi —
resize sớm phá enhancement. `max_canvas: 768` chặn bộ nhớ + kéo sensor ppi cao
(H ~930 ppi) gần về 500 ppi.

Cache: **1 file/ảnh nguồn** (không phải/cặp) — 13.629 ảnh (1 ảnh lỗi nguồn), mỗi
file ~1,5MB, tổng ~20GB tại `data/cached_stage1/{cache_key}_stage1.pt`.

```bash
python scripts/run_offline_preprocessing.py --device cuda   # full dataset (~25-40 phút trên A4000)
python scripts/run_offline_preprocessing.py --limit 50      # smoke test
```

Verify trực quan (overlay minutiae/orientation lên ảnh thật):
```bash
python scripts/verify_coarsenet.py --device cuda
```

### ⚠️ Giới hạn đã biết của CoarseNet trên dataset này

- **Sensor H over-detect nghiêm trọng**: 410 minutiae/ảnh (kỳ vọng 40-150) — ảnh
  sensor H đã bị nhị phân hoá sẵn (không phải grayscale liên tục), lệch domain so
  với dữ liệu CoarseNet được train gốc. Ảnh hưởng ~8,9% số cặp training (H làm
  nguồn). E/G cũng hơi cao (183, 194) nhưng nhẹ hơn nhiều.
- **`evaluate.py` tính minutiae F1 trên ảnh 256×256** (đã resize) trong khi
  CoarseNet cần ~500 ppi → cả target thật lẫn ảnh sinh đều ra 0 minutiae phát
  hiện được ở bước eval. Đã sửa lỗi decode (dùng đúng `extract_minutiae()` thay
  vì threshold trên `minutiae_map` đã bilinear-upsample), nhưng vấn đề resolution
  vẫn còn — **Minutiae F1 hiện chưa đáng tin, chỉ nên nhìn Orientation RMSE + ảnh
  thật**.

---

## 4. Phase 1 — Train VAE (`scripts/train_vae.py`)

VAE nhỏ (~15,6M tham số, `src/models/vae.py`), grayscale 256×256 ↔ latent 4×32×32
(nén 8×). Train trên **ảnh đơn** đã de-duplicate (12.286 ảnh unique, không lặp qua
~72k cặp).

```bash
python scripts/train_vae.py --epochs 20                          # từ đầu
python scripts/train_vae.py --epochs 20 --resume weights/vae_fingerprint.pt  # tiếp tục
```

- Loss: reconstruction (MSE) + KL (`kl_weight=1e-4`, giữ nhỏ để ảnh sắc nét)
- LR: AdamW 1e-4 + `CosineAnnealingLR` (giảm dần về `eta_min=1e-6`)
- Checkpoint: `weights/vae_latest.pt` (mỗi epoch) + `weights/vae_fingerprint.pt` (best loss)

**Kết quả hiện tại**: 20 epoch (resume từ checkpoint DMCS epoch 0), best loss
**0,0249** (giảm từ 0,177 ban đầu ~7 lần), reconstruction MSE cuối 0,0185.

---

## 5. Phase 2 — Train MM-DiT (`scripts/train_mmdit.py`)

`src/models/mm_dit.py::DualStreamMMDiT` — 350,86M tham số (`hidden_size=768,
depth=16, num_heads=12`). VAE freeze hoàn toàn.

### Composite loss (`src/losses/loss_builder.py`)

```
L_total = L_Diff + w_id(t,epoch)·L_Identity + w_ori(t,epoch)·L_Orient
```

- **L_Diff**: MSE giữa velocity dự đoán và velocity thật (rectified flow,
  `src/models/flow_matching.py`)
- **L_Identity**: cosine distance embedding giữa ảnh sinh và ảnh nguồn — dùng
  **DMD** (Dense Minutia Descriptor, IJCB 2024, Tsinghua, Apache-2.0), vendor tại
  `src/losses/dmd/`, weight `weights/dmd.pt` (checkpoint gốc train trên NIST SD14,
  strict-load 429/429 tensor khớp). Wrapper `DMDEmbeddingExtractor`
  (`src/losses/identity_loss.py`) pool có mask từ dense descriptor DMD thành
  1 vector 12-d, chuẩn hoá L2. Thay thế `AFRNetEmbeddingExtractor` gốc (random-init,
  chưa từng có pretrained weight thật).
- **L_Orient**: coherence giữa orientation field trích từ ảnh sinh (Sobel gradient)
  và `S_aligned` target — cả hai đều ép **fp32** bất kể AMP autocast
  (`src/preprocessing/orientation.py`) do bug đã fix (xem mục 6).
- Warmup: `L_Diff` một mình 5 epoch đầu, sau đó `L_Identity`/`L_Orient` ramp
  tuyến tính lên trong 5 epoch tiếp.

### Lịch learning rate

```python
if step < warmup_steps:       # 1000 step
    lr = peak_lr * step/warmup_steps
else:                          # cosine decay về sàn 5% peak_lr
    lr = min_lr_ratio + (1-min_lr_ratio) * 0.5 * (1+cos(π·progress))
```

```bash
python scripts/train_mmdit.py --epochs 10                        # từ đầu
python scripts/train_mmdit.py --epochs 10 --resume outputs/training/latest.pt
python scripts/train_mmdit.py --small                            # model nhẹ, test CPU
```

Checkpoint: chỉ giữ **2 file cố định** `outputs/training/latest.pt` (đầy đủ, để
resume) + `best.pt` (chỉ model+EMA, nhẹ hơn ~2×, để inference) — không tích luỹ
`epoch_N.pt` như thiết kế ban đầu (đĩa server dùng chung, hay đầy).

**Kết quả hiện tại (10 epoch, LR cosine decay, sensor H giữ nguyên trong data)**:

| Epoch | Train Loss | Val Loss | Ghi chú |
|---|---|---|---|
| 1 | 0,5535 | 0,5053 | |
| 3 | 0,4702 | 0,4920 | vượt best của lần chạy LR-phẳng trước |
| 6 | 0,4520 | 0,4914 | |
| **7** | 0,4572 | **0,4902** | **best** — ID/Orient vừa kích hoạt, val vẫn cải thiện |
| 10 | 0,4804 | 0,4960 | |

So với lần chạy đầu (LR phẳng 1e-4 suốt, không decay): best cũ 0,4999 và **trôi
lên** sau epoch 5 khi Identity/Orient kích hoạt. Lần này thấp hơn (0,4902) và
không trôi lên khi kích hoạt đa mục tiêu — xác nhận LR phẳng là nguyên nhân chính
khiến lần trước chững.

---

## 6. Các bug đã gặp và cách sửa (quan trọng khi debug tiếp)

| Bug | Nguyên nhân | Sửa |
|---|---|---|
| `torch.amp.GradScaler` không tồn tại | API hợp nhất chỉ có từ torch≥2.3, môi trường chạy 2.1.0 | Dùng `torch.cuda.amp.GradScaler` |
| `L_Orient` ra NaN toàn bộ khi vừa kích hoạt (epoch warmup đầu tiên) | Epsilon `1e-8` trong `norm = sqrt(Vx²+Vy²+1e-8)` bị **underflow về 0 dưới fp16** (nhỏ hơn subnormal tối thiểu ~5,96e-8 của fp16) khi ảnh gần đồng nhất (early training) → `0/0=nan` | `src/preprocessing/orientation.py`: ép fp32 cho toàn bộ phép tính Sobel-gradient + chuẩn hoá, bọc `torch.autocast(enabled=False)` |
| Stage 1 cache có file `.pt` rỗng/hỏng (0 byte hoặc zip dở dang) | Process bị kill giữa chừng (session restart) đúng lúc đang `torch.save()` | Quét toàn bộ cache bằng `torch.load()` thử, xoá file lỗi, chạy lại preprocessing (tự skip phần đã cache tốt) |
| `dataset.require_cache=True` làm training crash khi gặp 1 ảnh nguồn hỏng thật | Không có bước lọc trước — `__getitem__` raise `FileNotFoundError` giữa DataLoader worker | `CrossSensorFingerprintDataset._drop_uncached()`: lọc sample thiếu cache **lúc khởi tạo dataset**, in cảnh báo, thay vì crash lúc training |
| Checkpoint MM-DiT tích luỹ không giới hạn (`epoch_N.pt` mỗi 5 epoch) | Thiết kế ban đầu không tính đến đĩa dùng chung, hay đầy (từng xuống còn 25GB) | Chỉ giữ `latest.pt` + `best.pt`; `best.pt` bỏ luôn optimizer/scheduler state (chỉ dùng inference) |
| Minutiae F1 luôn = 0 kể cả ảnh thật | `evaluate.py` threshold trực tiếp trên `minutiae_map` đã bilinear-upsample (làm loãng peak) + chạy CoarseNet trên ảnh 256px thay vì ~500ppi gốc | Đã sửa phần decode (dùng `extractor.extract_minutiae()` đúng cách — threshold ở độ phân giải gốc 1/8 + sub-cell offset); **phần resolution mismatch vẫn còn tồn đọng**, chưa sửa |

---

## 7. Evaluate (`scripts/evaluate.py`)

```bash
python scripts/evaluate.py --checkpoint outputs/training/best.pt --num_samples 16 --save_images
python scripts/evaluate.py --num_samples -1   # toàn bộ val set (8.276 sample, chậm)
```

Với mỗi sample: Euler sampling (`nfe_steps=20`, mặc định) từ noise → latent → VAE
decode → ảnh. So với ảnh target thật:

- **Orientation RMSE** (độ) — tin cậy được
- **Minutiae precision/recall/F1** — chưa tin cậy (xem mục 3/6)
- **PSNR, MSE** — tin cậy được, chỉ tương quan thô với chất lượng thị giác
- **Orientation cosine similarity** — dễ gây hiểu lầm (tính trên toàn ảnh gồm cả
  nền, giá trị cao không đồng nghĩa vân tay giống nhau)

**Kết quả gần nhất** (checkpoint `best.pt` epoch 7, LR cosine decay):

| Metric | Lần đầu (LR phẳng, ~random) | Lần này (LR decay, 10 epoch) |
|---|---|---|
| Orientation RMSE | 46,31° | **44,98°** |
| PSNR | 13,36 dB | **13,68 dB** |
| Minutiae F1 | 0,0 (không tin cậy) | 0,0 (không tin cậy) |

Đánh giá bằng mắt: ảnh sinh **đã có texture hướng vân mờ** (khác hẳn nhiễu thuần
tuý của lần đầu), nhưng chưa có core/delta rõ ràng, chưa đạt chất lượng vân tay
thật. RMSE ~45° vẫn gần mức ngẫu nhiên (chance-level ~45° do góc wrap 0-180°) —
cần train nhiều epoch hơn để hội tụ rõ rệt.

---

## 8. Cấu trúc thư mục

```
configs/default_config.yaml     Toàn bộ hyperparameter + đường dẫn dataset/weight
data/cached_stage1/             Cache Stage 1 (1 file/ảnh nguồn, ~20GB)
weights/
  coarsenet_pytorch.pt          CoarseNet converted từ Keras .h5
  vae_fingerprint.pt            VAE Phase 1 (best loss)
  dmd.pt                        DMD identity embedder (NIST SD14 pretrained)
outputs/
  training/{latest,best}.pt     Checkpoint MM-DiT (bounded, 2 file)
  eval/                         Kết quả evaluate.py (JSON + ảnh)
  coarsenet_check/               Ảnh verify CoarseNet
src/
  data/                         Dataset loader + SD302a inspector
  preprocessing/                CoarseNet extractor, Stage 1 offline cache, TPS (chưa hoàn thiện), orientation utils
  models/                       VAE, MM-DiT (+ RoPE2D), flow matching, baseline ControlNet
  losses/                       Identity (DMD), orientation, composite loss builder
  evaluation/                   Metric functions
  ablation/                     Harness chạy ablation study
scripts/                        Entry point cho từng bước (xem các mục trên)
tests/                          pytest — 9/10 pass (1 fail: môi trường thiếu opencv-contrib cho TPS, không liên quan)
analysis_results.md             Nhật ký phân tích/quyết định chi tiết theo thời gian
```

---

## 9. Việc còn lại / hướng tiếp theo

| # | Việc | Ưu tiên |
|---|---|---|
| 1 | Train MM-DiT nhiều epoch hơn (LR decay đã đúng, chỉ cần thêm thời gian) | 🔴 |
| 2 | Sửa resolution mismatch trong `evaluate.py` để Minutiae F1 đáng tin | 🟠 |
| 3 | Xử lý sensor H over-detect minutiae (loại khỏi dataset, hoặc threshold riêng) | 🟡 |
| 4 | Minutiae matching giữa 2 sensor để TPS `is_aligned` thực sự `True` (hiện luôn `False`) | 🟡 |
| 5 | Cài `opencv-contrib-python` để test TPS pass đủ 10/10 | 🟢 |
