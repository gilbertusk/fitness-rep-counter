# Model Card — pengenal latihan & penghitung repetisi

Status per 2026-10-02: **pengenal latihan terlatih dan dievaluasi** (run `data/runs/20261002-1640`,
dilatih oleh pemilik proyek); **penghitung repetisi belum dievaluasi** (menunggu label manusia).
Semua angka disalin dari laporan yang dihasilkan skrip; bagian ⏳ belum ada laporannya.

## 1. Ringkasan

| | Pengenal latihan | Penghitung repetisi |
|---|---|---|
| Tugas | 22 kelas latihan dari window 2 detik keypoint | jumlah & waktu rep, latihan apa pun |
| Jenis | 1D-CNN (`ml/src/repcount/models/temporal.py`) + baseline gradient boosting | algoritma tanpa training (`app/src/core/counting/genericCounter.js`) |
| Ukuran | 63 702 parameter, ≈ 0,28 MB ONNX (opset 18, float32) | – |
| Input | `[batch, 30, 47]`: 30 frame @ 15 fps × (13 landmark × x, y, visibility + 8 sudut) — [`FEATURES.md`](FEATURES.md) | 33 landmark mentah per frame |
| Output | skor 22 kelas → `classifier.js`: smoothing EMA + "tidak yakin" | jumlah rep, waktu tiap rep, status bergerak/diam |
| Runtime | `onnxruntime-web` 1.23.2 di browser (WebGPU, fallback WASM) | JS murni, ≈ 0,1 ms per frame |
| Keypoint | MediaPipe `pose_landmarker_lite` float16 v1 — sama untuk training dan browser | sama |

## 2. Data latih

- [Workout & Exercise Video Dataset](https://www.kaggle.com/datasets/ziya07/workout-and-exercise-video-dataset)
  (Kaggle, CC0): 652 video, 651 terbaca, 22 kelas, campuran potongan YouTube dan rekaman HP.
- Split per video **dan** per grup near-duplicate, seed 42: train 453 / val 99 / test 99
  ([`reports/00-data/split.md`](../reports/00-data/split.md)). Test hanya dipakai sekali, di
  `classifier_report`.
- Tidak seimbang: 7 (plank) sampai 62 (barbell biceps curl) video per kelas; ditangani dengan bobot
  kelas dan dilaporkan dengan macro-F1.
- Augmentasi saat training: flip kiri↔kanan (p = 0,5), jitter 0,01 unit torso, skala waktu 0,8–1,2.
- Kualitas pose: 97,7 % frame terdeteksi; terburuk decline bench press 87,6 %
  ([`reports/00-data/pose_quality.md`](../reports/00-data/pose_quality.md)).
- Window yang benar-benar dipakai (window dengan > 30 % frame hilang dibuang): train 2761, val 738,
  test 606 dari **93** dari 99 video test ([`classifier.md`](../reports/01-classifier/classifier.md)).
- Label penghitung repetisi: **hanya dari manusia**, ±104 video val/test, definisi di
  [`labels/README.md`](../labels/README.md). ⏳ belum dibuat.

## 3. Kelas

barbell_biceps_curl · bench_press · chest_fly_machine · deadlift · decline_bench_press ·
hammer_curl · hip_thrust · incline_bench_press · lat_pulldown · lateral_raise · leg_extension ·
leg_raises · plank · pull_up · push_up · romanian_deadlift · russian_twist · shoulder_press ·
squat · t_bar_row · tricep_dips · tricep_pushdown

## 4. Metrik

| Metrik | Nilai | Sumber |
|---|---|---|
| Macro-F1 / akurasi, level video, 93 video test | **0,822 / 81,7 %** (baseline gradient boosting 0,756 / 80,6 %) | [`classifier.md`](../reports/01-classifier/classifier.md) |
| Macro-F1 / akurasi, level window, 606 window test | 0,761 / 78,7 % (baseline 0,723 / 77,4 %) | [`classifier.md`](../reports/01-classifier/classifier.md) |
| Macro-F1 / akurasi, level window, val (pemilihan epoch) | 0,795 / 77,8 % | output `repcount.models.temporal` |
| F1 per kelas terburuk (level video) | biceps curl 0,50 · romanian deadlift 0,50 · decline bench 0,57 · chest fly 0,60 · bench press 0,67 | [`classifier.md`](../reports/01-classifier/classifier.md) |
| Ambang "tidak yakin" (disapu di val, akurasi ≥ 90 %) | keyakinan 0,65 (cakupan 74 %) · selisih 0,45 (cakupan 73 %) | [`labels.json`](../app/models/labels.json) |
| ONNX vs PyTorch, 100 window | beda maks 5,7e-06 (toleransi 1e-04) | [`classifier.md`](../reports/01-classifier/classifier.md) |
| MAE & OBO penghitung vs baseline naive-peaks, per kelas (test) | ⏳ | `reports/03-rep-counter/reps_*_test.csv` (dibuat oleh `evalReps.js`) → `rep_counter.md` |
| Latensi pengenal per window | 0,08 ms (Python CPU) · 1,1 ms p50 di browser (4 vCPU, tanpa GPU) | [`classifier.md`](../reports/01-classifier/classifier.md), [`performance.md`](../reports/05-performance/performance.md) |
| Latensi penghitung per frame | 0,085–0,28 ms p50 (tergantung VM) | [`performance.md`](../reports/05-performance/performance.md) |

## 5. Batasan

- **Sudut kamera:** pose 2D dari satu kamera. Latihan yang gerakannya searah pandang kamera (squat
  dan push-up dari depan) memberi sinyal lemah; aturan form butuh kamera dari samping.
- **Satu orang:** MediaPipe dipakai dengan `numPoses = 1`; orang lain di frame bisa "mencuri" pose.
- **Pencahayaan & oklusi:** gelap, cahaya dari belakang, pakaian longgar, atau beban/mesin yang menutupi
  sendi menurunkan deteksi (posisi berbaring dan mesin paling buruk di dataset ini).
- **Kelas lemah & test kecil:** lima kelas terburuk di atas; bench press paling sering tertukar dengan
  decline bench press. Test hanya 1–9 video per kelas, jadi F1 per kelas kasar (plank = 1 video).
  Dugaan awal bahwa russian twist (rotasi) akan sulit **tidak** terbukti di test (F1 1,0 — tetapi hanya ≤ 2 video).
- **Domain:** video gym dari YouTube dan HP; perekaman di rumah, sudut tak biasa, atau tubuh di luar
  sebaran dataset belum diuji.
- **Penghitung:** rep dilaporkan ±0,2 s lebih awal dari tanda manusia. Periode hanya terbaca sampai
  2 s (setengah buffer 4 s); gerakan yang sangat dangkal atau lebih lambat dari itu belum teruji dan
  bisa terlewat.

## 6. Penggunaan yang tidak disarankan

- **Bukan alat medis, fisioterapi, atau rehabilitasi.** Peringatan form adalah heuristik geometri
  yang ambangnya belum divalidasi; jangan dipakai untuk mendiagnosis cedera atau menggantikan pelatih.
- Bukan untuk penilaian, kompetisi, atau keputusan tentang seseorang (asuransi, kerja, sekolah).
- Bukan untuk memantau orang tanpa persetujuannya — meskipun video tidak pernah meninggalkan perangkat.

## 7. Cara memperbarui

```bash
python -m repcount.evaluation.classifier_report --run $RUN   # → reports/01-classifier/ + ambang di labels.json
python -m repcount.export.onnx --run $RUN                    # → app/models/exercise_classifier.onnx
node tools/eval/evalReps.js --counter generic --split test --final   # → reports/03-rep-counter/*.csv
```

Lalu salin angka dari laporan ke tabel §4 di sini dan di README root, dengan tautan ke laporannya.
