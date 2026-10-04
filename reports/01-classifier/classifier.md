# Pengenal jenis latihan (22 kelas)

Perintah: `python -m repcount.models.baseline` · `python -m repcount.models.temporal` · `python -m repcount.evaluation.classifier_report` · `python -m repcount.export.onnx`

Window 30 frame @ 15 fps (stride 15), 47 fitur per frame — spesifikasi: `docs/FEATURES.md`.

> **Test set dipakai sekali saja**, oleh skrip ini, di akhir. Pemilihan model dan semua ambang diputuskan dari split validasi.

Window: train 2761, val 738, test 606 (window dengan > 30% frame hilang dibuang).

## Hasil di test set

### Level window

| Model | Akurasi | Macro-F1 | n window |
|---|---:|---:|---:|
| baseline (gradient boosting) | 0.774 | 0.723 | 606 |
| temporal (1D-CNN) | 0.787 | 0.761 | 606 |

### Level video (rata-rata probabilitas seluruh window video)

| Model | Akurasi | Macro-F1 | n video |
|---|---:|---:|---:|
| baseline (gradient boosting) | 0.806 | 0.756 | 93 |
| temporal (1D-CNN) | 0.817 | 0.822 | 93 |

Model terbaik menurut macro-F1 level video: **temporal (1D-CNN)**.

## F1 per kelas (level video, model terbaik)

| Kelas | F1 |
|---|---:|
| barbell_biceps_curl | 0.500 |
| bench_press | 0.667 |
| chest_fly_machine | 0.600 |
| deadlift | 0.800 |
| decline_bench_press | 0.571 |
| hammer_curl | 1.000 |
| hip_thrust | 1.000 |
| incline_bench_press | 0.889 |
| lat_pulldown | 0.933 |
| lateral_raise | 0.923 |
| leg_extension | 1.000 |
| leg_raises | 0.750 |
| plank | 1.000 |
| pull_up | 0.857 |
| push_up | 1.000 |
| romanian_deadlift | 0.500 |
| russian_twist | 1.000 |
| shoulder_press | 0.667 |
| squat | 0.857 |
| t_bar_row | 1.000 |
| tricep_dips | 0.800 |
| tricep_pushdown | 0.769 |

Lima kelas terburuk: barbell_biceps_curl (0.50), romanian_deadlift (0.50), decline_bench_press (0.57), chest_fly_machine (0.60), bench_press (0.67).

## Kelas yang paling sering tertukar

| Sebenarnya | Diprediksi | Jumlah | % dari kelas |
|---|---|---:|---:|
| barbell_biceps_curl | chest_fly_machine | 2 | 22% |
| barbell_biceps_curl | leg_raises | 2 | 22% |
| bench_press | decline_bench_press | 2 | 25% |
| barbell_biceps_curl | decline_bench_press | 1 | 11% |
| barbell_biceps_curl | tricep_pushdown | 1 | 11% |

Dugaan penyebab dikaitkan dengan `reports/00-data/pose_quality.md`: kelas dengan deteksi pose terburuk (decline_bench_press 87,6%, romanian_deadlift 88,3%, bench_press 91,7%) adalah latihan berbaring dan mesin, di mana pose 2D dari satu kamera sulit membedakan sudut bangku.

![Confusion matrix](figures/confusion_matrix.png)

## Ambang "tidak yakin"

Dipilih dari split **validasi**: cut-off terkecil yang membuat akurasi prediksi yang dipertahankan mencapai 90%.

| Ambang | Nilai | Cakupan | Akurasi saat dipertahankan |
|---|---:|---:|---:|
| confidence | 0.65 | 74% | 0.902 |
| margin | 0.45 | 73% | 0.910 |

## Model di browser

- Berkas: `app/models/exercise_classifier.onnx`, **270 KB**, opset 18
- ONNX vs PyTorch pada 100 window: beda maks **5.72e-06** (toleransi 1e-04)
- Waktu inferensi CPU per window: median **0.08 ms**, p95 0.09 ms (200 kali)
