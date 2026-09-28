# Tahap 1 — Pengenal Jenis Latihan (22 kelas)

> Salin seluruh isi file ini sebagai prompt ke AI.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md`** (§3, §4 struktur folder — **wajib diikuti**, §6, §8) dan `reports/00-data/pose_quality.md`. Tahap 0 sudah selesai: keypoint ada di `data/keypoints/`, split di `ml/splits/split_v1.json`. Kerjakan **hanya Tahap 1**.

## Tujuan

Model kecil yang menebak jenis latihan dari jendela keypoint ±2 detik, berjalan di browser, dengan evaluasi jujur dan fitur yang **identik** antara Python dan JS.

## Lokasi file tahap ini

```
docs/FEATURES.md
ml/src/repcount/features/      features.py, windows.py, golden.py
ml/src/repcount/models/        baseline.py, temporal.py
ml/src/repcount/evaluation/    classifier_report.py
ml/src/repcount/export/        onnx.py
ml/tests/features/ , ml/tests/models/ , ml/tests/export/
app/src/core/features/         features.js
app/src/core/classify/         classifier.js
app/src/adapters/              onnxClassifier.js
app/tests/unit/features/ , app/tests/unit/classify/
app/tests/fixtures/            features_golden.json
app/models/                    exercise_classifier.onnx, labels.json
reports/01-classifier/         classifier.md, figures/
```

## Tugas

1. **Spesifikasi fitur** — tulis `docs/FEATURES.md` dulu, baru kode.
   - Resample ke **15 fps**, window **30 frame (2 detik)**, stride 15 frame.
   - Normalisasi per frame: pusat = titik tengah pinggul, skala = jarak bahu–pinggul rata-rata. Pakai 12 landmark tubuh utama + hidung.
   - Fitur per frame: koordinat ter-normalisasi (x, y), visibility, 8 sudut sendi (siku, bahu, pinggul, lutut kiri/kanan).
   - NaN: interpolasi linear jika gap ≤ 5 frame; window dengan > 30% frame hilang dibuang saat training dan ditandai "tidak yakin" saat inferensi.
   - Augmentasi flip kiri-kanan saat training (tukar landmark kiri/kanan + negasi x).

2. **Implementasi fitur ganda + parity test**
   - `repcount.features.features` dan `app/src/core/features/features.js` mengikuti spesifikasi yang sama.
   - `python -m repcount.features.golden` membuat `app/tests/fixtures/features_golden.json`.
   - Test di **kedua** sisi membandingkan dengan golden file, toleransi 1e-4.

3. **Dataset window** — `repcount.features.windows`: dari `.npz` + split → array window per split, dengan `video_id` per window.

4. **Baseline** — `python -m repcount.models.baseline`
   - Ringkasan per window (mean, std, min, max, range, frekuensi dominan tiap sudut) + `HistGradientBoostingClassifier` (class weight balanced).

5. **Model temporal** — `python -m repcount.models.temporal` (PyTorch, tambahkan ke extra `train` di `pyproject.toml`)
   - 1D-CNN kecil (< 1 MB) atau GRU kecil; pilih dan jelaskan alasannya.
   - Class-weighted loss, augmentasi (flip, jitter, skala waktu 0,8–1,2×), early stopping pada **macro-F1 val**, seed tetap.
   - Checkpoint + config ke `data/runs/<timestamp>/`.

6. **Evaluasi** — `python -m repcount.evaluation.classifier_report`
   → `reports/01-classifier/classifier.md` + `reports/01-classifier/figures/confusion_matrix.png`
   - Pada **test set** (sekali, di akhir): accuracy, macro-F1, F1 per kelas; level window dan level video (rata-rata probabilitas).
   - Tabel baseline vs temporal; 5 pasangan kelas paling sering tertukar + dugaan penyebab (kaitkan dengan laporan kualitas pose).
   - Ambang "tidak yakin" dipilih dari val set.

7. **Ekspor ke browser** — `python -m repcount.export.onnx` → `app/models/exercise_classifier.onnx` + `app/models/labels.json`
   - Verifikasi ONNX vs PyTorch pada 100 window, beda maks < 1e-4.
   - `app/src/core/classify/classifier.js`: pasca-proses murni (softmax, EMA antar window, ambang "tidak yakin"), di-unit-test.
   - `app/src/adapters/onnxClassifier.js`: loader `onnxruntime-web` (CDN), tanpa logika.

## Kriteria selesai

- [ ] Semua file baru berada di lokasi yang tercantum di atas; tidak ada folder baru di luar §4.
- [ ] Parity test fitur lulus di pytest **dan** `npm test`.
- [ ] `reports/01-classifier/classifier.md` lengkap + perintah reproduksi.
- [ ] Ukuran model ONNX dan waktu inferensi per window (CPU) dicatat.
- [ ] Test set hanya dipakai sekali (tulis eksplisit di laporan).
- [ ] Coverage ≥ 80% untuk modul fitur & klasifikasi (Python dan JS); `ruff check ml` bersih.
- [ ] `ml/README.md` diperbarui dengan perintah tahap ini; `docs/PLAN.md` diperbarui (Status + Log Keputusan).

## Laporan akhir

Angka utama (macro-F1 baseline vs temporal, akurasi level video), kelas terburuk, ukuran model, dan keputusan yang butuh persetujuan saya (mis. menggabung incline/decline bench press). Lalu berhenti.
