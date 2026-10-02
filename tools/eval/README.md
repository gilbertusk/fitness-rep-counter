# tools/eval — evaluasi penghitung repetisi

Memutar setiap video berlabel **frame demi frame** ke sebuah counter (seperti di browser, tanpa
melihat masa depan) lalu membandingkannya dengan label manusia (`labels/rep_labels.csv`).

```bash
python -m repcount.export.keypoints_json                          # sekali: data/keypoints_json/
node tools/eval/evalReps.js --counter generic --split val          # → reports/03-rep-counter/reps_generic_val.csv
node tools/eval/evalReps.js --counter generic --split val --grid   # tuning, hanya di val
node tools/eval/evalReps.js --counter generic --split test --final # sekali, di akhir
python -m repcount.evaluation.rep_plots --traces data/runs/rep_traces_generic_test.json
node tools/eval/evalReps.js --bench                                # ms per frame, tanpa dataset
```

| Counter | Isi |
|---|---|
| `generic` | `genericCounter.js`, sinyal PCA online |
| `generic-angle` | `genericCounter.js`, sinyal sudut sendi paling bervariasi |
| `naive-peaks` | puncak sudut paling bervariasi, parameter tetap — **batas bawah** |
| `threshold` | `thresholdCounter.js` dengan sudut manual; hanya squat & push-up, kelas lain **N/A** |

**Metrik** (`lib/repMetrics.js`): MAE, OBO (|prediksi − label| ≤ 1) dan MAE ternormalisasi
dirata-rata **per video**; precision, recall, dan latensi dihitung **per rep** (pencocokan greedy,
toleransi 1 detik). Ringkasan selalu dua kali: semua video, dan tanpa video ambigu. Plank tidak
dinilai di sini — itu durasi tahan, bukan repetisi.

**Membaca latensi:** counter generik menghitung saat sinyal melewati 70% jalan kembali ke posisi awal,
jadi latensinya **negatif ±0,2 s** — rep muncul sebelum orangnya selesai kembali. Itu disengaja
(aplikasi terasa responsif), bukan kesalahan.

**Penjaga:** `--split test` ditolak tanpa `--final`; `--grid` hanya di val; video berlabel tanpa
keypoint dicetak sebagai "missing", tidak dilewati diam-diam.

Logika murni ada di `lib/` dan diuji oleh `tests/` lewat `npm test`.
