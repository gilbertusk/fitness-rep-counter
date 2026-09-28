# Tahap 3 — Penghitung Repetisi Generik

> Salin seluruh isi file ini sebagai prompt ke AI.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md`** (§3, §4 struktur folder — **wajib diikuti**, §6, §8), `docs/FEATURES.md`, dan `labels/README.md`. Tahap 0–2 sudah selesai; `labels/rep_labels.csv` berisi label manusia. Kerjakan **hanya Tahap 3**.

## Tujuan

Satu penghitung repetisi untuk semua latihan **tanpa threshold per latihan**, berjalan **streaming** (frame demi frame, tanpa melihat masa depan), dan dibandingkan jujur dengan penghitung threshold yang ada.

## Lokasi file tahap ini

```
ml/src/repcount/export/        keypoints_json.py
ml/src/repcount/evaluation/    rep_plots.py
app/src/core/counting/         genericCounter.js, naivePeakCounter.js
app/src/core/geometry/         oneEuroFilter.js (jika dipakai)
app/tests/unit/counting/
tools/eval/                    evalReps.js, README.md
tools/eval/lib/                repMetrics.js, counters.js (registry nama → counter)
tools/eval/tests/
reports/03-rep-counter/        rep_counter.md, reps_<counter>_<split>.csv, figures/
```

## Tugas

1. **Ekspor keypoint untuk Node** — `python -m repcount.export.keypoints_json` → `data/keypoints_json/<video_id>.json` (hanya video berlabel).

2. **Harness evaluasi** — `node tools/eval/evalReps.js --counter <nama> --split val|test`
   - Memberi frame satu per satu ke counter (**streaming**), mencatat jumlah akhir dan waktu setiap rep.
   - Metrik per video & per kelas: **MAE**, **OBO** (|pred − gt| ≤ 1), MAE ternormalisasi, dan precision/recall per rep (matching greedy, toleransi 1 detik) + latensi deteksi.
   - Output ke `reports/03-rep-counter/reps_<counter>_<split>.csv` + ringkasan terminal.
   - Logika metrik di `tools/eval/lib/repMetrics.js` (murni, di-unit-test).

3. **Baseline**
   - `threshold`: `thresholdCounter.js` untuk kelas yang punya konfigurasi (squat, push-up); kelas lain dilaporkan **N/A**, bukan nol.
   - `naive-peaks`: `naivePeakCounter.js`, deteksi puncak sederhana pada sudut paling bervariasi — batas bawah.

4. **Counter generik** — `app/src/core/counting/genericCounter.js` (murni, immutable, streaming)
   - Sinyal: proyeksi fitur ter-normalisasi ke komponen utama yang diperbarui online, **atau** sudut sendi dengan variansi terbesar di jendela berjalan — bandingkan keduanya, dokumentasikan.
   - Smoothing EMA / One Euro (parameter didokumentasikan).
   - Periode dari autokorelasi buffer berjalan ±4 detik; puncak dengan jarak min ≈ 0,6 × periode dan prominence adaptif; hysteresis.
   - Keadaan diam: tidak menghitung jika amplitudo di bawah ambang.
   - Tuning **hanya di val** (grid kecil, dicatat); test dijalankan sekali di akhir.

5. **Laporan** — `reports/03-rep-counter/rep_counter.md`
   - Tabel counter × (MAE, OBO, precision/recall) keseluruhan & per kelas pada **test**.
   - Grafik (`python -m repcount.evaluation.rep_plots` → `figures/`): sinyal + rep terdeteksi + label untuk 1 video berhasil dan 2 gagal.
   - Analisis kegagalan dikaitkan dengan `reports/00-data/pose_quality.md`; perintah reproduksi setiap angka.

## Test (node:test)

- Sinus sintetis N periode → N rep untuk beberapa frekuensi & amplitudo; sinus + noise → tanpa hitungan ganda; sinyal datar → 0; perubahan tempo → tetap terhitung.
- State tidak dimutasi.
- Metrik: MAE, OBO, matching rep dengan contoh kecil yang dihitung manual.

## Kriteria selesai

- [ ] Semua file di lokasi yang tercantum; tidak ada folder baru di luar §4.
- [ ] `npm test` lulus, coverage ≥ 80% untuk `genericCounter.js` dan `repMetrics.js`.
- [ ] `reports/03-rep-counter/rep_counter.md` lengkap dengan perbandingan baseline pada test.
- [ ] Counter generik < 1 ms per frame di Node (diukur & dicatat).
- [ ] `tools/eval/README.md` dan `docs/PLAN.md` diperbarui (Status + Log Keputusan).

## Laporan akhir

MAE/OBO generik vs baseline, kelas terbaik & terburuk, rekomendasi latihan yang layak di demo vs ditulis sebagai keterbatasan. Lalu berhenti.
