# Rep Counter: Penghitung Repetisi & Koreksi Form Olahraga

Web app yang menghitung repetisi **squat** dan **push-up** serta memberi koreksi form secara real-time dari webcam atau video.
Semua proses berjalan **di browser** (MediaPipe Pose + WebAssembly/WebGL), jadi video tidak pernah dikirim ke server.

> 🔗 Demo: _(isi setelah deploy)_ · 🎬 GIF demo: _(taruh di sini)_

## Cara kerja

```
Webcam / video ─► MediaPipe Pose (33 keypoint) ─► pilih sisi tubuh paling terlihat
   ─► hitung sudut sendi (lutut / siku) ─► state machine UP ⇄ DOWN (hysteresis) ─► jumlah repetisi
   └► aturan form (kemiringan punggung, garis badan) ─► peringatan
```

| Latihan | Sudut penghitung | Turun | Naik | Aturan form |
|---|---|---|---|---|
| Squat | pinggul–lutut–pergelangan kaki | ≤ 95° | ≥ 160° | torso miring > 45° → "Punggung terlalu membungkuk" |
| Push-up | bahu–siku–pergelangan tangan | ≤ 90° | ≥ 155° | bahu–pinggul–kaki < 160° → "Jaga badan tetap lurus" |

Gerakan yang turun tapi tidak cukup dalam tidak dihitung dan diberi feedback "Kurang dalam".

## Menjalankan secara lokal

```bash
npm start          # buka http://localhost:5173
npm test           # unit test (node:test, tanpa dependency)
npm run test:coverage
```

Kamera hanya bisa diakses dari `localhost` atau HTTPS.

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
- [ ] Rekam 20–30 video uji (form benar/salah), lalu tuning threshold
- [ ] Evaluasi di [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html): MAE jumlah repetisi, dicantumkan di README
- [ ] Tambah latihan: lunge, bicep curl, shoulder press
- [ ] Smoothing sudut (EMA) & indikator kepercayaan pose
- [ ] Umpan balik suara (Web Speech API)
- [ ] Deploy (Vercel / GitHub Pages) + GitHub Actions menjalankan `npm test`
- [ ] Klasifikasi jenis latihan otomatis (22 kelas): kode, spesifikasi fitur & parity test selesai; training + laporan evaluasi belum

## Keterbatasan

- Threshold masih nilai awal dan perlu dituning dengan data nyata.
- Kamera sebaiknya menghadap tubuh dari samping; sudut kamera frontal membuat sudut sendi kurang akurat.
- Hanya satu orang per frame.
