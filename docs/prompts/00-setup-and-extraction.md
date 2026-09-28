# Tahap 0 — Restrukturisasi Repo, Setup & Ekstraksi Keypoint

> Salin seluruh isi file ini sebagai prompt ke AI.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md` terlebih dahulu**, terutama §3 (keputusan), §4 (struktur folder & konvensi — **wajib diikuti persis**), dan §6 (aturan kerja). Kerjakan **hanya Tahap 0**.

## Konteks

- Kode sekarang masih datar: `index.html`, `styles.css`, `src/`, `tests/` di root.
- Web app memakai MediaPipe `pose_landmarker_lite`. Ekstraksi offline **wajib** memakai model yang sama.
- Dataset **sudah ada** di `data/workout-videos/` di dalam repo (22 folder kelas, 652 video, ±4,6 GB). Folder `data/` **tidak boleh** masuk git. Jika suatu saat perlu download ulang:
  `kaggle datasets download ziya07/workout-and-exercise-video-dataset -p data/workout-videos --unzip`.
- Python 3.11 tersedia. GPU GTX 1650 tidak wajib untuk tahap ini.

## Tugas

### 1. Restrukturisasi repo ke struktur §4 (commit terpisah: `refactor: restructure repo layout`)

Pakai `git mv` agar riwayat file terjaga. Pemetaan:

| Lama | Baru |
|---|---|
| `index.html`, `styles.css` | `app/index.html`, `app/styles.css` |
| `src/main.js`, `src/config.js` | `app/src/main.js`, `app/src/config.js` |
| `src/core/angles.js` | `app/src/core/geometry/angles.js` |
| `src/core/repCounter.js` | `app/src/core/counting/thresholdCounter.js` |
| `src/core/pose.js` | `app/src/core/form/measure.js` |
| `src/core/exercises.js` | `app/src/core/exercises.js` |
| `tests/*.test.js` | `app/tests/unit/<subfolder yang sama dengan modulnya>/…` |
| `tests/push-up_17.mp4` | `app/tests/fixtures/videos/push-up_17.mp4` |

- Pecah `main.js` minimal: pemanggilan MediaPipe ke `app/src/adapters/poseLandmarker.js`, render kerangka/teks ke `app/src/ui/overlay.js`, `main.js` hanya merangkai. **Perilaku tidak boleh berubah.**
- Perbarui semua import dan `package.json`: `start` melayani folder `app/`, `test` menjalankan semua `*.test.js` di `app/tests/unit` dan `tools/` (cek versi Node untuk sintaks glob yang didukung).
- **Langkah pertama sebelum commit apa pun:** pastikan `.gitignore` berisi `data/`, lalu cek `git status` tidak menampilkan satu pun file dari `data/`.
- `.gitignore` lengkap: `node_modules/`, `coverage/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `data/`, `*.npz`, `.env`.
- Buat `README.md` pendek (≤ 15 baris) di `app/`, `ml/`, `tools/`, `labels/`, `reports/`.
- Verifikasi: `npm test` lulus dan `npm start` masih bisa menghitung push-up pada video fixture.

### 2. Paket Python `repcount`

- `ml/pyproject.toml`: paket `repcount` (layout `src/`), dependency di-pin: `mediapipe`, `opencv-python-headless`, `numpy`, `pandas`; extra `dev`: `pytest`, `pytest-cov`, `ruff`.
- `.venv` di root repo, `pip install -e "ml[dev]"`. Jika `.venv` yang ada rusak/tidak lengkap, hapus dan buat ulang.
- `ml/src/repcount/config.py`: `DATA_DIR` = folder `data/` di root repo (bisa di-override dengan env `FITNESS_DATA_DIR`), turunan path (`VIDEOS_DIR`, `KEYPOINTS_DIR`, `POSE_MODELS_DIR`, `RUNS_DIR`), serta konstanta ekstraksi (`MAX_FPS = 30`, `MAX_SIDE_PX = 640`).
- `repcount.data.download_model`: unduh `pose_landmarker_lite.task` ke `POSE_MODELS_DIR` (URL sama dengan `app/src/config.js`).

### 3. Manifest — `repcount.data.manifest`

- Pindai dataset → `KEYPOINTS_DIR/manifest.csv`: `video_id`, `label` (snake_case, mis. `pull Up` → `pull_up`), `path`, `ext`, `fps`, `n_frames`, `duration_s`, `width`, `height`, `size_bytes`, `error`.
- Video yang gagal dibuka dicatat di `error`, skrip tidak crash.

### 4. Deteksi near-duplicate — `repcount.data.dedup`

- dHash 64-bit (implementasi sendiri, OpenCV/numpy) dari frame di 10%, 50%, 90% durasi.
- Rata-rata jarak Hamming ≤ ambang (mulai 6, dokumentasikan) → `group_id` sama (union-find). Tulis `group_id` ke manifest.

### 5. Ekstraksi keypoint — `repcount.data.extract`

- Pose Landmarker mode `VIDEO`, `num_poses=1`, confidence sama dengan web app.
- Sampling ke maks `MAX_FPS`; resize sisi terpanjang ke `MAX_SIDE_PX` sebelum inferensi.
- Output: `KEYPOINTS_DIR/<label>/<video_id>.npz` berisi
  `landmarks` float32 `(T, 33, 4)` = x, y, z, visibility (**NaN** jika pose tidak terdeteksi), `world_landmarks` float32 `(T, 33, 3)`, `timestamps_ms` int64 `(T,)`, `fps_effective`, `width`, `height`, `label`, `video_id`.
- Resumable, paralel (`--workers`, default core − 2), progress sederhana, error ke `KEYPOINTS_DIR/errors.csv`. CLI: `--limit`, `--workers`, `--overwrite`.

### 6. Laporan kualitas pose — `repcount.evaluation.pose_quality`

→ `reports/00-data/pose_quality.md` + `pose_quality.csv`
- Per kelas: jumlah video, total durasi, % frame dengan pose terdeteksi, rata-rata visibility per kelompok sendi (bahu, siku, pergelangan tangan, pinggul, lutut, pergelangan kaki), sisi tubuh dominan.
- Urutkan dari kelas terburuk; 3–5 temuan singkat bahasa Indonesia.

### 7. Split — `repcount.data.split`

→ `ml/splits/split_v1.json` + `reports/00-data/split.md`
- Train/val/test = 70/15/15 per kelas, stratified, **berdasarkan `group_id`**, seed 42.
- Kelas < 10 video: usahakan minimal 1 video di val dan test; catat pengecualian.

## Test (pytest, di `ml/tests/`, struktur mencerminkan `src/repcount/`)

- Normalisasi nama label.
- Indeks frame untuk sampling fps (60 → 30, 24 → 24, 29.97).
- dHash & jarak Hamming (gambar sintetis identik vs berbeda); union-find pengelompokan.
- Split: tidak ada grup di dua split, proporsi mendekati target, deterministik.
- Agregasi laporan kualitas dari array landmarks sintetis dengan NaN.

Logika murni dipisahkan dari I/O agar bisa di-test tanpa video.

## Kriteria selesai

- [ ] Pohon folder sesuai §4 PLAN (tampilkan output `tree` / `git ls-files` di laporan), tidak ada file lepas di root di luar peta.
- [ ] `npm test` lulus setelah restrukturisasi; app tetap berfungsi.
- [ ] `pytest ml/tests --cov=repcount` lulus, coverage ≥ 80% untuk modul logika; `ruff check ml` bersih.
- [ ] Uji cepat `python -m repcount.data.extract --limit 5` berhasil, lalu ekstraksi penuh selesai (catat durasi).
- [ ] Jumlah `.npz` = jumlah video berhasil; kegagalan tercatat.
- [ ] `reports/00-data/*` dan `ml/splits/split_v1.json` di-commit.
- [ ] README root: bagian "Struktur repo" (pohon ringkas) dan "Pipeline data" (perintah menjalankan ulang Tahap 0).
- [ ] `docs/PLAN.md`: Status Tahap 0 ✅ dan Log Keputusan diisi (ambang dedup, jumlah duplikat, durasi ekstraksi, kelas bermasalah).

## Laporan akhir

Ringkasan: pohon folder baru, angka utama (video berhasil/gagal, % deteksi kelas terburuk & terbaik, jumlah grup duplikat, jumlah per split), dan **keputusan yang butuh persetujuan saya** (mis. menggabung/membuang kelas). Lalu berhenti — jangan mulai Tahap 1.
