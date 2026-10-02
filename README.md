# Rep Counter: Penghitung Repetisi & Koreksi Form Olahraga

Web app yang menghitung repetisi untuk **berbagai latihan dengan satu penghitung generik**, memberi koreksi form
untuk 6 latihan + mode plank, dan (setelah model dilatih) mengenali latihannya otomatis — dari webcam atau
video. Semua proses berjalan **di browser** (MediaPipe Pose + WebAssembly), jadi video tidak pernah dikirim ke server.

> 🔗 Demo: _(isi setelah deploy)_ · 🎬 GIF demo: _(taruh di sini)_

**Status jujur:** belum ada angka akurasi. Pengenal latihan belum dilatih (Tahap 1) dan penghitung repetisi belum
dievaluasi dengan label manusia (Tahap 2–3), jadi semua latihan ditandai **eksperimental** di app dan latihan
dipilih manual. Rinciannya di `docs/PLAN.md` §8.

## Cara kerja

```
Webcam / video ─► MediaPipe Pose (33 keypoint, mentah)
   ├─► penghitung repetisi generik (streaming, tanpa threshold per latihan)  ─► jumlah rep
   ├─► window fitur 2 detik ─► pengenal latihan ONNX (setelah dilatih)        ─► "squat (93%)"
   └─► One Euro filter ─► aturan form per latihan (debounce 0,5 s)           ─► peringatan
                       └► timer tahan plank
sesi: Siap ─► Mengenali ─► Menghitung ─► Istirahat (diam > 3 s) ─► …
```

Aturan form dan batasannya: [`docs/FORM_RULES.md`](docs/FORM_RULES.md). Penghitung dan alasan desainnya:
`app/src/core/counting/genericCounter.js`.

## Menjalankan secara lokal

```bash
npm start              # buka http://localhost:5173 — kamera hanya bisa diakses dari localhost atau HTTPS
npm test               # unit test (node:test)
npm run test:coverage
npm run test:e2e       # Playwright; sekali dulu: npm install && python -m repcount.data.download_model
```

## Privasi: video tidak meninggalkan perangkat

Diverifikasi oleh `app/tests/e2e/smoke.spec.js`: selama app memproses video, **setiap** request yang dibuat halaman
dicatat, dan test gagal bila ada yang bukan `GET`, membawa body, atau menuju host selain app itu sendiri, CDN
MediaPipe (`cdn.jsdelivr.net`), dan bucket model pose (`storage.googleapis.com`). Yang diunduh hanya kode dan
model; frame video diproses di memori browser dan tidak pernah dikirim.

## Struktur repo

Detail & konvensi: [`docs/PLAN.md` §4](docs/PLAN.md).

```
app/        # 🌐 web app yang di-deploy (index.html, src/core|adapters|ui, tests/)
ml/         # 🧠 paket Python `repcount`: ekstraksi keypoint, split, (nanti) training & evaluasi
tools/      # 🛠️ alat bantu developer (labeler, evaluasi Node, benchmark)
labels/     # ✍️ label buatan manusia
reports/    # 📊 hasil evaluasi, nomor = tahap (00-data/, …)
docs/       # 📚 PLAN.md & prompt per tahap
data/       # 💾 dataset & turunan — di-.gitignore, tidak pernah di-commit
```

## Pipeline data (Tahap 0)

Butuh Python 3.11. Semua data ada di `data/` (override: env `FITNESS_DATA_DIR`).

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; Linux/macOS: .venv/bin/activate
pip install -e "ml[dev]"

# dataset (sekali saja, ±4,6 GB)
kaggle datasets download ziya07/workout-and-exercise-video-dataset -p data/workout-videos --unzip

python -m repcount.data.download_model        # pose_landmarker_lite.task → data/pose_models/
python -m repcount.data.manifest              # → data/keypoints/manifest.csv
python -m repcount.data.dedup                 # group_id near-duplicate → manifest.csv
python -m repcount.data.extract               # → data/keypoints/<kelas>/<video_id>.npz (resumable, --workers N)
python -m repcount.evaluation.pose_quality    # → reports/00-data/pose_quality.md/.csv
python -m repcount.data.split                 # → ml/splits/split_v1.json + reports/00-data/split.md

pytest ml/tests --cov=repcount && ruff check ml
```

## Pengenal jenis latihan (Tahap 1 — kode siap, model belum dilatih)

Spesifikasi fitur ada di [`docs/FEATURES.md`](docs/FEATURES.md) dan diimplementasikan **dua kali** —
Python untuk training, JS untuk browser — dijaga oleh parity test di kedua sisi (toleransi 1e-4):

```bash
python -m repcount.features.golden   # regenerasi fixture parity (tidak butuh dataset)
pytest ml/tests/features -q          # parity sisi Python
npm test                             # parity sisi JS
```

Perintah training, evaluasi, dan ekspor ada di [`ml/README.md`](ml/README.md); semuanya butuh
`data/keypoints/` dari Tahap 0. **Belum ada angka akurasi di repo ini karena model belum pernah
dilatih** — lihat `docs/PLAN.md` §8 untuk sisa pekerjaannya.

## Label repetisi (Tahap 2 — alat siap, menunggu label manusia)

Ground truth jumlah dan waktu tiap rep untuk ±104 video val/test, sebagai bahan evaluasi penghitung
repetisi. Labelnya **hanya dari manusia**; repo ini menyediakan alatnya:

```bash
python -m repcount.labels.select     # → labels/to_label.csv (butuh data/ dari Tahap 0)
npx serve tools/labeler              # alat labeling di browser, dikendalikan keyboard
python -m repcount.labels.validate   # cek labels/rep_labels.csv
```

Definisi satu repetisi per gerakan dan panduan kerja (±2 jam): [`labels/README.md`](labels/README.md).

## Penghitung repetisi generik (Tahap 3 — kode siap, belum dievaluasi)

Satu counter untuk semua latihan tanpa threshold per latihan, streaming frame demi frame
(`app/src/core/counting/genericCounter.js`), plus harness yang membandingkannya dengan dua baseline.
**Belum ada angka akurasi**: evaluasinya butuh label Tahap 2. Yang sudah terukur hanya kecepatannya —
di bawah 0,25 ms per frame (p95) di Node.

```bash
node tools/eval/evalReps.js --bench                          # kecepatan, tanpa dataset
node tools/eval/evalReps.js --counter generic --split val    # setelah label & keypoint ada
```

Petunjuk lengkap: [`tools/eval/README.md`](tools/eval/README.md).

## Roadmap

- [x] MVP: squat & push-up, penghitung + 2 aturan form, unit test
- [x] Penghitung repetisi generik untuk semua latihan (Tahap 3) — kode & harness; **evaluasi menunggu label**
- [x] App Tahap 4: sesi otomatis, aturan form 6 latihan + plank, One Euro filter, indikator kualitas pose,
      suara, riwayat set, kamera depan/belakang, smoke test Playwright
- [ ] Label repetisi manusia (Tahap 2) → evaluasi MAE/OBO penghitung (Tahap 3)
- [ ] Training pengenal latihan (Tahap 1) → deteksi otomatis aktif di app
- [ ] Video uji form benar/salah berlabel, untuk mengukur akurasi aturan form
- [ ] Deploy (GitHub Pages) + GitHub Actions + benchmark FPS di perangkat nyata (Tahap 5)

## Keterbatasan

- **Belum ada angka akurasi** untuk pengenal latihan, penghitung repetisi, maupun aturan form (lihat Status di atas).
- Ambang aturan form adalah titik awal yang belum divalidasi; aturan deadlift hanya proksi 2D (`docs/FORM_RULES.md`).
- Aturan form squat, push-up, curl, deadlift, dan plank butuh kamera dari samping; app memberi petunjuk bila
  kamera tampak dari depan.
- Hanya satu orang per frame.
- Di perangkat tanpa GPU sungguhan (WebGL perangkat lunak), MediaPipe dijalankan di CPU — terukur ±35 ms per
  frame di mesin uji 4-core; perangkat yang lebih lemah akan lebih lambat.
