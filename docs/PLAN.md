# Rencana Proyek: Rep Counter → Pelatih Olahraga Multi-Latihan

Dokumen ini adalah **sumber kebenaran** untuk semua pekerjaan (manusia maupun AI).
Setiap prompt di `docs/prompts/` mengasumsikan dokumen ini sudah dibaca.

## 1. Tujuan

Mengembangkan rep counter (sekarang: squat & push-up dengan threshold manual) menjadi sistem yang
**mengenali 22 jenis latihan secara otomatis, menghitung repetisi, dan memberi koreksi form**,
berjalan sepenuhnya di browser, dengan **setiap klaim didukung angka evaluasi**.

Kalimat target di README:
> Mengenali 22 latihan dengan akurasi X% (macro-F1 Y), menghitung repetisi dengan MAE Z rep,
> berjalan N FPS di browser, dan video tidak pernah meninggalkan perangkat.

## 2. Dataset

- Sumber: Kaggle `ziya07/workout-and-exercise-video-dataset` (CC0), 652 video, 22 kelas, ±4,9 GB.
- Label: **hanya jenis latihan**. Tidak ada jumlah rep dan tidak ada label form → label rep dibuat sendiri (Tahap 2).
- Distribusi tidak seimbang: barbell biceps curl 62, bench press 61, push-up 56, lat pulldown 51,
  tricep Pushdown 50, lateral raise 37, incline bench press 33, deadlift 32, squat 29, chest fly machine 28,
  pull Up 26, leg extension 25, t bar row 21, leg raises 21, tricep dips 20, hammer curl 19, hip thrust 18,
  shoulder press 17, romanian deadlift 14, russian twist 13, decline bench press 12, plank 7.
- Format campuran: `.mp4` kecil (kemungkinan potongan YouTube) dan `.MOV` besar (rekaman HP).

## 3. Keputusan yang sudah dikunci

| Topik | Keputusan | Alasan |
|---|---|---|
| Lokasi data | Semua data di **satu folder `data/` di root repo**, di-`.gitignore` (bisa di-override dengan env `FITNESS_DATA_DIR`) | Tidak terpencar, tapi video 4,6 GB & data turunan tidak masuk git |
| Model pose offline | **`pose_landmarker_lite` yang sama dengan web app** (`app/src/config.js`) | Konsistensi train/serve: model dilatih di keypoint yang sama dengan yang dilihat browser |
| Lingkungan Python | venv `.venv/` di root repo; paket `repcount` di `ml/` di-install editable (`pip install -e ml[dev]`), versi di-pin | Reproducible, dipanggil dengan `python -m repcount...` |
| Struktur folder | Mengikuti §4 **persis**; folder baru hanya dengan persetujuan | Repo rapi & mudah dibaca |
| Pembagian data | Per **video** (bukan per frame/window), stratified, seed tetap, disimpan di `ml/splits/split_v1.json` (di-commit) | Mencegah kebocoran data |
| Near-duplicate | Dicek di Tahap 0; video yang hampir identik diberi `group_id` yang sama dan split mengikuti grup | Potongan dari video YouTube yang sama bisa bocor antar split |
| Fitur model | Satu spesifikasi fitur, diimplementasikan di Python **dan** JS, dijaga dengan **parity test** (golden file) | Hasil di browser harus sama dengan hasil evaluasi |
| Penghitung repetisi | Hidup di **JS** (`src/core/`), dievaluasi dengan Node secara **streaming frame-by-frame** | Satu sumber kebenaran; evaluasi meniru kondisi real-time |
| Inferensi klasifikasi di browser | ONNX → `onnxruntime-web` | Ringan, didukung PyTorch export |
| Metrik klasifikasi | Accuracy, macro-F1, F1 per kelas, confusion matrix; level window **dan** level video | Macro-F1 jujur untuk kelas tidak seimbang |
| Metrik repetisi | MAE, OBO (off-by-one accuracy), per kelas | Metrik standar (RepCount/TransRAC) |
| Plank | Mode khusus: timer tahan + cek garis badan, bukan repetisi | Latihan isometrik |

## 4. Arsitektur target

```
Video/webcam ─► MediaPipe Pose Lite (33 keypoint)
   ├─► features.js ─► classifier.onnx ─► "lat pulldown (0.93)"
   ├─► genericCounter.js (periodisitas, streaming) ─► jumlah rep
   └─► formRules (per latihan) ─► peringatan
```

### Struktur folder (wajib diikuti)

Prinsip: **tiga dunia terpisah jelas** — `app/` (yang dipakai user), `ml/` (yang melatih model), `tools/` (alat bantu developer). Hasil kerja manusia ada di `labels/`, hasil evaluasi di `reports/`, dokumentasi di `docs/`. Nomor pada `reports/` dan `docs/prompts/` = nomor tahap.

```
fitness-rep-counter/
├── README.md
├── package.json                     # task runner JS: start, test, lint
├── .gitignore                       # mengabaikan data/, .venv/, node_modules/, dll.
├── .github/workflows/ci.yml
│
├── data/                            # 💾 SEMUA data (di-.gitignore, tidak di-commit)
│   ├── workout-videos/<kelas>/      # dataset mentah (*.mp4, *.MOV)
│   ├── pose_models/                 # pose_landmarker_lite.task
│   ├── keypoints/<kelas>/           # <video_id>.npz + manifest.csv, errors.csv
│   ├── keypoints_json/              # untuk evaluasi di Node
│   └── runs/                        # <timestamp>/ checkpoint training; rep_traces_*.json (Tahap 3)
│
├── app/                             # 🌐 Web app — folder ini yang di-deploy
│   ├── index.html
│   ├── styles.css
│   ├── models/                      # exercise_classifier.onnx, labels.json
│   ├── src/
│   │   ├── main.js                  # entry point: hanya merangkai modul
│   │   ├── config.js                # URL model & library
│   │   ├── core/                    # logika MURNI (tanpa DOM/browser API), 100% di-test
│   │   │   ├── exercises.js         # katalog latihan
│   │   │   ├── geometry/            # angles.js, oneEuroFilter.js
│   │   │   ├── features/            # features.js (spesifikasi: docs/FEATURES.md)
│   │   │   ├── classify/            # classifier.js (pasca-proses prediksi)
│   │   │   ├── counting/            # thresholdCounter.js, genericCounter.js, holdTimer.js
│   │   │   ├── form/                # measure.js, rules/<latihan>.js
│   │   │   └── session/             # session.js (state machine sesi)
│   │   ├── adapters/                # pembungkus browser API: poseLandmarker.js, onnxClassifier.js, speech.js, storage.js
│   │   └── ui/                      # DOM & render: camera.js, overlay.js, panel.js
│   └── tests/
│       ├── unit/                    # mencerminkan struktur src/core/
│       ├── e2e/                     # Playwright smoke test
│       └── fixtures/                # features_golden.json, syntheticPose.js, videos/
│
├── ml/                              # 🧠 Pipeline Python
│   ├── pyproject.toml               # paket `repcount`, dependency di-pin
│   ├── splits/                      # split_v1.json (di-commit)
│   ├── src/repcount/
│   │   ├── config.py                # path ke data/, konstanta global
│   │   ├── data/                    # download_model, manifest, dedup, extract, split
│   │   ├── features/                # features, windows, golden
│   │   ├── models/                  # baseline, temporal
│   │   ├── evaluation/              # pose_quality, classifier_report, rep_plots
│   │   ├── export/                  # onnx, keypoints_json
│   │   └── labels/                  # select, validate
│   └── tests/                       # mencerminkan struktur src/repcount/
│
├── tools/                           # 🛠️ Alat bantu developer (tidak di-deploy)
│   ├── labeler/                     # alat labeling repetisi (web statis)
│   │   ├── index.html
│   │   ├── src/
│   │   └── tests/
│   ├── eval/                        # evaluasi counter di Node
│   │   ├── evalReps.js
│   │   ├── lib/repMetrics.js
│   │   └── tests/
│   ├── benchmark/                   # benchmark FPS di browser
│   └── checkStructure.js            # dipakai CI: menjaga struktur folder ini
│
├── labels/                          # ✍️ Label manusia (di-commit)
│   ├── README.md                    # definisi 1 repetisi & aturan labeling
│   ├── to_label.csv
│   ├── rep_labels.csv
│   └── rep_labels_recheck.csv       # label ulang ±10% untuk mengukur konsistensi (Tahap 2)
│
├── reports/                         # 📊 Hasil evaluasi (di-commit), nomor = tahap
│   ├── 00-data/                     # pose_quality.md/.csv, split.md
│   ├── 01-classifier/               # classifier.md, figures/
│   ├── 03-rep-counter/              # rep_counter.md, *.csv, figures/
│   └── 05-performance/              # performance.md
│
└── docs/                            # 📚 Dokumentasi
    ├── PLAN.md
    ├── FEATURES.md
    ├── FORM_RULES.md
    ├── MODEL_CARD.md
    └── prompts/                     # 00-…05-… prompt per tahap
```

### Konvensi penamaan

| Jenis | Aturan | Contoh |
|---|---|---|
| File & fungsi JS | camelCase | `genericCounter.js`, `updateCounter()` |
| Test JS | `<nama>.test.js`, lokasi mencerminkan `src/` | `app/tests/unit/counting/genericCounter.test.js` |
| Modul Python | snake_case | `pose_quality.py` |
| Test Python | `test_<nama>.py`, lokasi mencerminkan `src/repcount/` | `ml/tests/data/test_split.py` |
| Label kelas | snake_case, dari nama folder dataset | `barbell_biceps_curl`, `pull_up` |
| Laporan | `reports/<NN>-<tahap>/<nama>.md`, gambar di `figures/` | `reports/01-classifier/figures/confusion_matrix.png` |
| Dokumen | UPPER_SNAKE di `docs/` | `FORM_RULES.md` |

Aturan kerapian:
- Setiap folder tingkat atas (`app/`, `ml/`, `tools/`, `labels/`, `reports/`) punya `README.md` pendek (≤ 15 baris): isi folder + perintah utama.
- Tidak ada file lepas di root selain yang tercantum di pohon di atas.
- Skrip Python dijalankan sebagai modul (`python -m repcount.data.extract`), bukan file lepas.
- `app/src/core/` tidak boleh meng-import dari `adapters/` atau `ui/`; `ui/` tidak boleh berisi logika perhitungan.

## 5. Tahapan

| # | Tahap | Prompt | Output utama | Status |
|---|---|---|---|---|
| 0 | Setup & ekstraksi keypoint | `prompts/00-setup-and-extraction.md` | keypoint `.npz`, manifest, laporan kualitas pose, split | ✅ |
| 1 | Pengenal latihan (22 kelas) | `prompts/01-exercise-classifier.md` | model ONNX, laporan evaluasi, parity test fitur | 🚧 kode siap, **belum dilatih** (lihat §8) |
| 2 | Label repetisi (manusia + alat) | `prompts/02-rep-labeling.md` | alat labeling, `labels/rep_labels.csv` | 🚧 alat siap, menunggu label manusia (lihat §8) |
| 3 | Penghitung repetisi generik | `prompts/03-generic-rep-counter.md` | `genericCounter.js`, laporan MAE/OBO vs baseline | 🚧 kode & harness siap, **belum dievaluasi** (butuh label Tahap 2, lihat §8) |
| 4 | Integrasi web app | `prompts/04-web-integration.md` | demo: auto-detect + hitung + form + plank | ☐ |
| 5 | Siap dipamerkan | `prompts/05-ship.md` | deploy, CI, benchmark FPS, README final | ☐ |

Urutan wajib: 0 → 1 → (2 bisa paralel dengan 1) → 3 → 4 → 5.

## 6. Aturan kerja untuk semua tahap

1. **Satu tahap per sesi.** Jangan mengerjakan tahap berikutnya tanpa diminta.
2. **Test dulu** untuk logika murni (pytest / node:test). Target coverage ≥ 80% untuk `src/core/` dan `ml/` (kecuali skrip I/O tipis).
3. **Jangan mengarang angka.** Semua angka di README/laporan harus berasal dari skrip yang bisa dijalankan ulang, dan perintahnya dicantumkan.
4. **Jangan membuat label palsu.** Label repetisi hanya dari manusia.
5. **Data mentah dan turunan tidak masuk git.** Yang di-commit: kode, split, label, laporan, model kecil.
6. **Struktur folder** mengikuti §4. Butuh folder/file di luar peta? Tanya dulu, lalu perbarui peta di §4.
7. **Immutable & kecil:** fungsi < 50 baris, file < 400 baris, tanpa mutasi state di `src/core/`.
8. **Commit** dengan format conventional commits (`feat:`, `fix:`, `docs:`, `test:`, ...), satu commit per unit kerja yang lulus test.
9. Di akhir tahap: jalankan semua test, perbarui kolom **Status** di tabel §5 dan isi **Log Keputusan** di §8, lalu laporkan ringkasan + angka + hal yang perlu keputusan manusia.

## 7. Risiko yang diketahui

- Deteksi pose lemah pada posisi berbaring (bench press) dan latihan mesin (lat pulldown, chest fly). Diukur di Tahap 0.
- Kelas mirip: bench/incline/decline, biceps curl/hammer curl, deadlift/romanian deadlift.
- Plank hanya 7 video → mungkin digabung/ditandai "data tidak cukup".
- Russian twist berupa rotasi, sulit dengan sudut 2D → kemungkinan besar keterbatasan yang didokumentasikan.
- Video `.MOV` beresolusi tinggi → ekstraksi lambat; frame di-resize sebelum inferensi.

## 8. Log keputusan

_(Diisi setiap akhir tahap: tanggal, keputusan, alasan, angka penting.)_

### 2026-09-28 — Tahap 0: setup & ekstraksi keypoint

- **Dataset:** 652 video terpindai, 651 berhasil diekstrak. Gagal: `decline bench press/dbp_4.MOV`
  (89 MB, `moov atom not found` → file rusak/terpotong), tercatat di `manifest.csv` & `errors.csv`.
- **Ekstraksi:** `pose_landmarker_lite` float16 v1 (URL sama dengan `app/src/config.js`), mode VIDEO,
  `num_poses=1`, confidence 0,5 (default MediaPipe = yang dipakai web app), maks 30 fps, sisi terpanjang 640 px.
  Ekstraksi penuh 646 video: **2949 s (±49 menit)**, 6 worker di CPU (5 video uji `--limit 5`: 23 s).
  Sebagian waktu itu CPU juga dipakai proses dedup, jadi angka ini batas atas.
- **Near-duplicate:** dHash 64-bit pada frame 10/50/90%, **ambang 10** (bukan 6). Alasan: mulai 6, lalu
  sampel pasangan di rentang 6–10 dicek visual → semuanya orang & tempat yang sama (potongan dari video
  sumber yang sama); pasangan beda kelas pertama muncul di ambang 12. Pita hitam (letterbox) dipotong
  dulu karena sempat membuat klip vertikal beda kelas terlihat mirip. Hasil: **70 grup** berisi >1 video
  (**184 video**), grup terbesar 11 video; 538 grup total. Sensitivitas (video dalam grup >1):
  ambang 6 → 123, 8 → 160, 10 → 184, 12 → 196.
- **Split v1:** 70/15/15 per kelas berdasarkan `group_id`, seed 42 → train 453 / val 99 / test 99.
  Pengecualian: plank hanya 7 video (5/1/1).
- **Kualitas pose:** 97,7% frame terdeteksi. Terburuk decline_bench_press 87,6%, romanian_deadlift 88,3%,
  bench_press 91,7%. 8 video deteksi < 50% (bench_press 4, lat_pulldown 2, leg_extension 2); 3 video tanpa
  pose sama sekali (lat_pulldown_25, lat_pulldown_6: close-up punggung; leg_extension_14: hanya kaki).
  Pergelangan kaki paling sering tak terlihat (visibility rata-rata 0,49).
- **Kelas bermasalah (menunggu keputusan):** plank (7 video), decline_bench_press (11 video, 1 rusak),
  3 video tanpa pose.

### 2026-09-28 — Tahap 1: pengenal latihan (kode selesai, model belum dilatih)

**Status: 🚧 belum selesai.** Seluruh kode, test, dan spesifikasi sudah ada dan lulus, tetapi
**tidak ada model terlatih, tidak ada `app/models/`, dan tidak ada `reports/01-classifier/`**,
karena lingkungan tempat tahap ini dikerjakan tidak punya `data/keypoints/` maupun kredensial Kaggle.
Angka akurasi/macro-F1 tidak dikarang (§6.3) — semuanya terbit setelah perintah di `ml/README.md`
dijalankan di mesin yang punya data.

- **Spesifikasi fitur** (`docs/FEATURES.md`): resample 15 fps, window 30 frame / stride 15,
  13 landmark + 8 sudut = **47 nilai per frame**, normalisasi pusat pinggul & skala bahu–pinggul.
  Dua keputusan tambahan di luar prompt:
  1. **Koreksi rasio aspek** (`x *= width/height`) sebelum apa pun. MediaPipe menormalisasi `x` ke
     lebar dan `y` ke tinggi, jadi tanpa ini sudut sendi terdistorsi pada frame non-persegi —
     dan dataset ini campuran `.mp4` lanskap & `.MOV` portrait.
  2. `z` **tidak dipakai**: kedalaman `pose_landmarker_lite` tidak stabil justru pada kelas yang
     kualitas posenya sudah terburuk (berbaring & mesin, lihat laporan Tahap 0).
- **Parity Python↔JS:** lulus di kedua sisi, toleransi 1e-4. Golden file dibuat dari **urutan pose
  sintetis deterministik** (seed 42), bukan video dataset, supaya parity bisa diperiksa di CI tanpa
  unduhan 4,6 GB. Urutan itu sengaja memuat gap NaN pendek (diinterpolasi), gap panjang (dibiarkan,
  membuang 1 dari 2 window), frame dengan skala torso merosot, dan visibility rendah.
- **Model:** 1D-CNN **63.702 parameter (270 KB float32)**, dipilih atas GRU karena window panjangnya
  tetap 30 frame dan konvolusi diekspor ke ONNX sebagai node yang ditangani `onnxruntime-web` secara
  dapat diprediksi (ekspor GRU membawa loop). Baseline pembanding: ringkasan statistik window +
  `HistGradientBoostingClassifier` balanced.
- **Jalur ekspor terukur** (bobot **acak** — ini mengukur ekspor, bukan akurasi): satu berkas
  270 KB, opset 18, `max |onnx − torch| = 4,5e-08` atas 100 window, **0,08 ms/window** (median, CPU).
- **Dua bug ditemukan test:**
  1. macro-F1 dirata-ratakan hanya atas kelas yang muncul di split → kelas yang absen (plank punya
     1 video test) akan menggelembungkan angka utama. Kini selalu atas seluruh 22 kelas.
  2. `torch.onnx.export` default `external_data=True` membuang bobot ke `*.onnx.data`, sehingga
     `app/models/` akan berisi graph 24 KB tanpa bobot. Kini inline, dan ekspor menolak lanjut
     bila berkas pendamping muncul.
- **Test:** 176 pytest + 71 node:test lulus; `ruff check ml` bersih. Coverage `repcount` 88%
  (features.py 100%, seluruh modul Tahap 1 ≥ 80%); JS `features.js` & `classifier.js` 100% baris.
  `ml/tests/test_pipeline_stage1.py` menjalankan seluruh rantai CLI di atas keypoint sintetis.
- **Ambang "tidak yakin"** di `app/src/core/classify/classifier.js` masih **placeholder** dan
  ditandai demikian di sumbernya; `classifier_report` menuliskan nilai hasil sweep val ke
  `thresholds.json`, lalu `export.onnx` meneruskannya ke `app/models/labels.json`.

**Sisa pekerjaan Tahap 1** (di mesin yang punya data): jalankan 4 perintah di `ml/README.md`,
lalu isi `reports/01-classifier/classifier.md`, `app/models/`, dan perbarui Status di §5.

### 2026-10-02 — Tahap 2: alat & label repetisi (alat siap, menunggu label manusia)

**Status: 🚧 alat siap, menunggu label manusia.** Belum ada satu label pun — sesuai §6.4, label
hanya dari manusia. `labels/to_label.csv` juga **belum ada**: pemilihannya butuh ekstensi & durasi
per video, yang hanya tersimpan di `data/keypoints/manifest.csv`, dan lingkungan tempat tahap ini
dikerjakan tidak punya `data/`. Satu perintah di mesin yang punya data: `python -m repcount.labels.select`.

- **Definisi satu rep** (`labels/README.md`): satu siklus penuh, ditandai saat gerakan **kembali ke
  posisi awal** — momen yang sama dengan saat penghitung Tahap 3 menambah hitungan, sehingga waktu
  tanda bisa dibandingkan, bukan hanya jumlahnya. Ditulis untuk 22 kelas, plus aturan rep parsial,
  video terpotong, lengan bergantian (tiap lengan = 1 rep, dicatat `bergantian`), dan plank sebagai
  durasi tahan.
- **Konflik aturan, diputuskan:** §4 membatasi README folder ≤ 15 baris, tetapi prompt Tahap 2
  menaruh definisi 22 gerakan + panduan kerja di `labels/README.md`. Prompt Tahap 2 diikuti (lebih
  spesifik), daripada membuat berkas baru di luar peta.
- **Pemilihan video:** maks 5 per kelas dari val + test, serakah dengan prioritas **grup
  near-duplicate baru** (dua klip dari satu sumber = orang & tempat yang sama, membuang waktu
  pelabel) → ekstensi baru → durasi terjauh. Generator acak per kelas, jadi daftar satu kelas tidak
  berubah bila kelas lain berubah. Video tanpa pose sama sekali dikeluarkan. Perkiraan: ±104 video
  (plank hanya 2, decline_bench_press / romanian_deadlift / russian_twist masing-masing 4).
- **Berkas baru di peta §4:** `labels/rep_labels_recheck.csv`, untuk sesi cek ulang yang disarankan
  prompt Tahap 2. Subset ±10%-nya dipilih deterministik di alat dari hash `video_id`, jadi tidak
  perlu berkas daftar terpisah.
- **Perkiraan waktu melabel: ±2 jam** + ±15 menit cek ulang. Rata-rata video 7,8 detik
  (`reports/00-data/pose_quality.csv`), jadi rekaman mentah ±14 menit.
- **Alat diuji di Chromium sungguhan** dengan Playwright (24/24): pencocokan folder, semua tombol,
  langkah per frame (33,4 ms pada 29,97 fps), bertahan setelah muat ulang, isolasi mode cek ulang,
  tanpa error console. **CSV hasil export lolos `repcount.labels.validate` tanpa diubah.**
  Catatan: Chromium bawaan Playwright tidak punya H.264, jadi alur lengkap diuji dengan salinan
  WebM dari video fixture; jalur galat diuji dengan MP4 aslinya.
- **Tiga bug ditemukan oleh pengujian:** status "N dari M video ditemukan" langsung terhapus saat
  video pertama dibuka (folder yang salah tidak akan ketahuan); `.MOV` HEVC dan `.mp4` hasil
  konversinya memetakan ke `video_id` yang sama dan yang terakhir terbaca menang (kini `.mp4`
  diutamakan); dan contoh `--video-id push-up_17` di dokumentasi Tahap 1 salah (yang benar
  `push_up_17`).
- **Risiko yang belum bisa diuji:** `.MOV` ponsel dengan codec HEVC mungkin tidak bisa diputar di
  browser. Alat menampilkan panduan konversi `ffmpeg` bila itu terjadi.
- **Test:** 233 pytest + 100 node:test lulus; `ruff check ml` bersih. Coverage modul label Python
  97%, `tools/labeler/src/labels.js` 100% baris.

**Sisa pekerjaan Tahap 2** (manusia, di mesin yang punya data): jalankan `select`, label ±104 video
di alat, export → `labels/rep_labels.csv`, lalu `validate`. Minimal 2 hari kemudian: sesi cek ulang
→ `validate --agreement`.

### 2026-10-02 — Tahap 3: penghitung repetisi generik (kode & harness siap, belum dievaluasi)

**Status: 🚧 belum dievaluasi.** Dikerjakan atas permintaan langsung **sebelum Tahap 1 dan 2 selesai**,
menyimpang dari urutan wajib §5. Akibatnya: tidak ada label (Tahap 2) dan tidak ada keypoint di
lingkungan ini, jadi **belum ada tuning di val, belum ada angka MAE/OBO, dan belum ada
`reports/03-rep-counter/`**. Semua angka di bawah berasal dari data sintetis atau pengukuran
kecepatan — tidak satu pun berkata apa-apa tentang akurasi di dataset asli.

- **Desain counter generik** (`app/src/core/counting/genericCounter.js`): fitur per frame dari
  `features.js` (x/y 13 landmark + 8 sudut) → resample 15 fps → EMA → window berjalan 4 s → satu
  sinyal → mesin zona dengan hysteresis (30% / 70% rentang). Rep dihitung saat sinyal **kembali ke
  sisi awalnya**, sejalan dengan konvensi label. **Penghitungan tidak menunggu periode**: video di
  dataset ini rata-rata 7,8 s dan sering hanya 1–3 rep, jadi counter yang butuh dua siklus untuk
  "mengunci" akan kehilangan rep pertama di hampir setiap video. Periode autokorelasi hanya mengatur
  jarak minimum (0,6 × periode) untuk ayunan dangkal; ayunan penuh (≥ 85% rentang) selalu dihitung.
- **Dua sinyal, sesuai prompt:** PCA online (iterasi pangkat, *warm start*, tanda diselaraskan) dan
  sudut sendi paling bervariasi (ganti kanal hanya bila menang 1,5×). Setiap sampel, **seluruh
  window diproyeksikan ulang** dengan definisi sinyal saat ini, sehingga rentang, periode, dan sisi
  awal selalu menggambarkan satu sinyal yang konsisten. Pilihan di antara keduanya menunggu val.
- **Empat cacat desain ditemukan oleh test sintetis dan sapuan acak 300 set, lalu diperbaiki:**
  1. Vektor gerak yang tidak di-*center* diproyeksikan ke sumbu PCA yang berputar pada derau →
     **diam terhitung 3 rep**.
  2. Sisi awal ("home") ditentukan dari satu sampel berderau → kini rata-rata 0,5 s pertama.
  3. EMA 0,35 meredam rep 0,8 s ke 64% amplitudo (vs 90% untuk rep 2 s), sehingga rep cepat tampak
     dangkal → kini 0,5 (81% vs 96%), dihitung dari respons frekuensi filter, bukan disetel ke test.
  4. Home yang terkunci dari jitter tepat di atas ambang diam mengikat gerakan asli yang jauh lebih
     besar → kini ditinjau ulang bila rentang tumbuh > 2× **sebelum** ada rep; setelah satu rep
     terhitung, home dibekukan (peninjauan di tengah set terbukti menghilangkan rep).
  Kedua regression test terakhir dibuktikan **gagal tanpa perbaikannya**.
- **Hasil sintetis setelah perbaikan:** 300/300 tepat pada sinyal acak; 160/160 pada aliran pose
  squat dan curl (hanya lengan) acak, untuk kedua sinyal.
- **Kecepatan (diukur, Node 22, container 4-core, 3× ulang):** generic-PCA rata-rata **0,07–0,11 ms**,
  p95 0,16–0,24 ms per frame; generic-angle 0,02–0,03 ms. Memenuhi < 1 ms. Maks sesekali ±2 ms
  (jeda GC/JIT). Jalur PCA sempat 3–4× lebih lambat (alokasi 60 baris per sampel dan satu perkalian
  titik konstan yang dihitung ulang per baris) — diperbaiki tanpa mengubah perilaku.
- **Waktu rep dilaporkan ±0,2 s lebih awal** daripada tanda manusia, karena rep dihitung saat sinyal
  melewati 70% jalan kembali. Disengaja (aplikasi terasa responsif); latensi di laporan akan negatif.
- **Baseline:** `naive-peaks` (parameter tetap, batas bawah) dan `threshold` (hanya squat & push-up;
  kelas lain dilaporkan **N/A**, tidak pernah nol). Temuan: siku lurus di dekat 180° hanya bisa
  terbaca lebih kecil saat ada derau, sehingga sebarannya miring dan lebar (163,6°–178,9° pada jitter
  0,002) — naive-peaks menghitung rep saat diam, dan di data sintetis menghitung berlebih ±1,6×.
- **Dua bug sambungan ditemukan oleh uji rantai end-to-end** (npz → JSON → Node → grafik): ekspor
  membuang seluruh frame bila satu landmark saja NaN (di data sintetis: semua frame), dan JS membaca
  `null` dari JSON sebagai 0 — landmark hilang akan terbaca sebagai titik di pojok gambar.
- **Penjaga kejujuran di harness:** `--split test` ditolak tanpa `--final`; `--grid` hanya di val
  (18 setelan untuk generic, 12 untuk naive-peaks); video berlabel tanpa keypoint dicetak, tidak
  dilewati diam-diam.
- **Peta §4 diperbarui:** `app/tests/fixtures/syntheticPose.js` (generator pose sintetis bersama untuk
  test counter dan harness) dan `data/runs/rep_traces_*.json` (jejak sinyal untuk grafik).
- **Test:** 245 pytest + 169 node:test lulus; `ruff check ml` bersih. Coverage `genericCounter.js` &
  `repMetrics.js` 100% baris.

**Sisa pekerjaan Tahap 3** (setelah `labels/rep_labels.csv` ada, di mesin yang punya data):
`keypoints_json` → `evalReps --grid` untuk `generic` dan `generic-angle` di val → pilih sinyal &
setelan → `evalReps` keempat counter di val → **sekali** `--split test --final` → `rep_plots` →
tulis `reports/03-rep-counter/rep_counter.md`.
