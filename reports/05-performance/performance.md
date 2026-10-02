# Performa (Tahap 5)

Diukur dengan `tools/benchmark/` — pipeline app yang sebenarnya, bukan simulasi. Cara mengukur &
definisi tiap angka: `tools/benchmark/README.md`.

## Ringkasan

| Perangkat | FPS end-to-end | Pose p50 / p95 | Core p50 / p95 | Status |
|---|---:|---:|---:|---|
| Container cloud, VM A — 4 vCPU, **tanpa GPU** (headless Chromium 141) | 22–23 | 39.7 / 54.0 ms | 0.105 / 0.195 ms | ✅ diukur (§1) |
| Container cloud, VM B — 4 vCPU, tanpa GPU, **model terlatih** | 9.9–11.5 | 74–93 / 118–127 ms | 0.26–0.50 / 0.64–0.91 ms | ✅ diukur (§1b) |
| Laptop | – | – | – | ⏳ belum diukur — perlu pemilik proyek |
| HP (Android, Chrome) | – | – | – | ⏳ belum diukur — perlu pemilik proyek |

**Bacaan:** hampir seluruh waktu per frame habis di MediaPipe Pose (≈ 99,7 %). Logika kita sendiri
(fitur + penghitung + aturan form + sesi) ≈ 0,1 ms per frame, jadi optimisasi berikutnya ada di sisi
pose (GPU delegate, resolusi input), bukan di `app/src/core/`. Container ini tidak punya GPU dan
WebGL-nya emulasi (SwiftShader), sehingga app memilih delegate **CPU** — kemungkinan besar ini batas bawah;
laptop dengan GPU nyata diharapkan lebih cepat, tapi **itu belum diukur**.

## 1. Container cloud (headless) — diukur 2026-10-02

Perintah (pengenal = arsitektur temporal asli dengan **bobot acak**, karena model terlatih belum ada —
hanya latensinya yang bermakna):

```bash
npm install --no-save onnxruntime-web@1.23.2          # agar runtime ONNX tersedia lokal
python - <<'PY'                                       # model acak 22 kelas → scratch/
import json, torch; from pathlib import Path
from repcount.models.temporal import build_model; from repcount.export.onnx import export_model
labels = [...]                                        # 22 label dari app/src/core/exercises.js (CATALOG)
torch.manual_seed(0); export_model(build_model(len(labels)), Path("scratch/random_model.onnx"))
PY
node tools/benchmark/headless.js --repeat 5 --model scratch/random_model.onnx --labels scratch/random_labels.json
```

Hasil persis seperti dicetak alat:

### Headless Chromium (CPU) — 2026-10-02

- Browser: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.7390.37 Safari/537.36`
- CPU logis: 4 · Delegate MediaPipe: **CPU** · WebGL: `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)`
- Video: push-up_17.webm (640×360, 3.0 s × 5 putaran)

| Ukuran | Nilai |
|---|---|
| FPS end-to-end (pose + core + overlay) | **23.0** (dibatasi FPS video ≈ 30) |
| Frame diproses | 354 (354 dengan pose) |
| Muat model pose (dari buka halaman) | 808.2 ms |
| Pose pertama terdeteksi (dari buka halaman) | 3424.6 ms |
| Muat pengenal latihan | 3072.7 ms |

| Tahap (ms per panggilan) | n | median | p95 | maks |
|---|---:|---:|---:|---:|
| Pose (MediaPipe detectForVideo) | 354 | 39.7 | 54.0 | 324.4 |
| Fitur (pushWindowStream) | 354 | 0.005 | 0.020 | 0.210 |
| Penghitung (updateGenericCounter) | 354 | 0.085 | 0.120 | 0.200 |
| Core total (stepFrame: fitur + counter + form + sesi) | 354 | 0.105 | 0.195 | 0.295 |
| Pengenal (ONNX, per window) | 14 | 0.800 | 7.9 | 20.4 |
| Frame utuh (pose + stepFrame + overlay) | 354 | 40.0 | 54.4 | 327.8 |

| Aset | File | Unduh (MB) | Setelah dekompresi (MB) |
|---|---:|---:|---:|
| Model pose (.task) | 1 | 5.78 | 5.78 |
| MediaPipe WASM | 1 | 11.76 | 11.76 |
| MediaPipe JS | 2 | 0.48 | 0.48 |
| ONNX Runtime WASM | 1 | 25.50 | 25.50 |
| ONNX Runtime JS | 2 | 0.12 | 0.12 |
| Model pengenal (.onnx) | 1 | 0.28 | 0.28 |
| Kode app (JS/CSS/HTML/JSON) | 24 | 0.08 | 0.08 |

> Pengenal: arsitektur temporal asli dengan BOBOT ACAK — hanya untuk mengukur latensi, skornya tidak berarti.
> Repetisi terhitung sepanjang benchmark (video diputar 5×): 5 — angka ini bukan evaluasi akurasi.

> App: buka halaman → "Model siap" (cache dingin): 868 ms.

Catatan kejujuran:

- Run kedua (`--repeat 2`, tanpa pengenal) memberi angka setara: 22,1 FPS, pose p50/p95 40,1 / 53,5 ms,
  core 0,085 / 0,135 ms. Sejalan dengan 21–24 FPS yang terbaca di app saat Tahap 4.
- FPS < 30 karena pose ≈ 40 ms > 33 ms anggaran per frame video 30 fps: frame yang datang saat pose
  masih bekerja dilewati (begitu juga di app — benchmark memakai loop `requestAnimationFrame` yang sama
  dengan `app/src/main.js`; versi awal dengan `requestVideoFrameCallback` hanya mendapat 15 FPS karena
  melewatkan frame dengan cara berbeda). Fitur di-resample ke 15 fps (`docs/FEATURES.md`), jadi
  22 FPS memberi margin di atas kebutuhan penghitung dan pengenal.
- "Maks" pose ≈ 350 ms = frame pertama (inisialisasi XNNPACK). p95 tidak memasukkannya secara berarti.
- Pengenal p50 0,8 ms vs p95 7,9 ms: panggilan pertama memanaskan runtime; n = 14 window saja.
- Waktu muat diukur dengan aset dari disk lokal (tanpa jaringan), jadi **tidak** mewakili koneksi
  nyata; di HP dengan 4G, unduhan aset (di bawah) yang menentukan.

## 1b. VM B dengan model terlatih — diukur 2026-10-02

Setelah container dimulai ulang, sesi berjalan di VM lain dengan spesifikasi tertulis sama (4 vCPU Xeon
2,8 GHz) tetapi ±2× lebih lambat. Pengenal kini **model terlatih** (`app/models/`, run 20261002-1640).

```bash
node tools/benchmark/headless.js --repeat 5        # model default: app/models/
```

Hasil run pertama, persis seperti dicetak alat:

### Headless Chromium (CPU) — 2026-10-02

- Browser: `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.7390.37 Safari/537.36`
- CPU logis: 4 · Delegate MediaPipe: **CPU** · WebGL: `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)`
- Video: push-up_17.webm (640×360, 3.0 s × 5 putaran)

| Ukuran | Nilai |
|---|---|
| FPS end-to-end (pose + core + overlay) | **11.1** (dibatasi FPS video ≈ 30) |
| Frame diproses | 172 (172 dengan pose) |
| Muat model pose (dari buka halaman) | 1422.3 ms |
| Pose pertama terdeteksi (dari buka halaman) | 4604.3 ms |
| Muat pengenal latihan | 4044.3 ms |

| Tahap (ms per panggilan) | n | median | p95 | maks |
|---|---:|---:|---:|---:|
| Pose (MediaPipe detectForVideo) | 172 | 77.6 | 117.9 | 522.5 |
| Fitur (pushWindowStream) | 172 | 0.010 | 0.140 | 0.245 |
| Penghitung (updateGenericCounter) | 172 | 0.255 | 0.545 | 0.785 |
| Core total (stepFrame: fitur + counter + form + sesi) | 172 | 0.498 | 0.912 | 1.2 |
| Pengenal (ONNX, per window) | 14 | 1.1 | 13.9 | 36.5 |
| Frame utuh (pose + stepFrame + overlay) | 172 | 78.6 | 119.1 | 527.1 |

| Aset | File | Unduh (MB) | Setelah dekompresi (MB) |
|---|---:|---:|---:|
| Model pose (.task) | 1 | 5.78 | 5.78 |
| MediaPipe WASM | 1 | 11.76 | 11.76 |
| MediaPipe JS | 2 | 0.48 | 0.48 |
| ONNX Runtime WASM | 1 | 25.50 | 25.50 |
| ONNX Runtime JS | 2 | 0.12 | 0.12 |
| Model pengenal (.onnx) | 1 | 0.28 | 0.28 |
| Kode app (JS/CSS/HTML/JSON) | 26 | 0.08 | 0.08 |

> Repetisi terhitung sepanjang benchmark (video diputar 5×): 5 — angka ini bukan evaluasi akurasi.

> App: buka halaman → "Model siap" (cache dingin): 4559 ms.

Dua run berikutnya: 11,5 dan 9,9 FPS; pose p50 74,4 / 92,8 ms; pengenal p50 1,1 ms di ketiganya.

**Model bukan penyebab perlambatan:** di VM yang sama **tanpa** memuat pengenal (`labels` kosong) hasilnya
10,0 FPS, pose p50 92,3 ms — sama dengan run bermodel. Penghitung di Node (`node tools/eval/evalReps.js
--bench`) juga lebih lambat di VM ini (generic rata-rata 0,12 ms vs 0,07–0,11 ms di VM A). Pengenal hanya
dipanggil sekali per detik (±1,1 ms), jadi bebannya < 0,2 % waktu. Pelajarannya: angka dari VM cloud
bersama bisa berbeda 2× antar-sesi — angka laptop dan HP (§3–4) yang menentukan.

## 2. Ukuran aset & muat pertama

| Aset | Ukuran file | Kapan diunduh |
|---|---:|---|
| Model pose `pose_landmarker_lite.task` | 5,78 MB | selalu |
| MediaPipe WASM (SIMD) + JS | 11,76 + 0,48 MB | selalu |
| Kode app (24 file JS/CSS/HTML/JSON) | 0,08 MB | selalu |
| ONNX Runtime WASM (`ort-wasm-simd-threaded.jsep.wasm`) + JS | 23,8–25,5 + 0,12 MB | hanya bila model pengenal ada |
| Model pengenal (temporal, 63 702 parameter) | 0,28 MB | hanya bila model pengenal ada |

Ini ukuran file **tanpa kompresi HTTP** (headless menyajikannya dari disk). CDN jsDelivr mengirim
dengan kompresi, jadi unduhan nyata lebih kecil — belum diukur karena CDN diblokir di container ini.
Ukur di laptop: DevTools → Network → kolom *Transferred*, "Disable cache", throttling "Fast 4G".

**Temuan:** bundle `ort.webgpu.min.mjs` menarik WASM JSEP ≈ 24 MB, dua kali WASM biasa (11,9 MB),
padahal model 63 ribu parameter berjalan 0,8 ms di CPU. Setelah ada model terlatih, pertimbangkan
`ort.wasm.min.mjs` (WASM saja) — keputusan ditunda sampai latensi WebGPU vs WASM bisa dibandingkan
dengan model sungguhan.

Muat pertama app (buka `app/` → "Model siap", cache dingin, aset lokal): 822–868 ms di VM A tanpa model
(2 run); 3,8–4,6 s di VM B dengan model terlatih (3 run) — termasuk memuat ONNX Runtime ±25 MB.

## 3. Laptop — diisi pemilik proyek

`npx --yes serve -l 5174 .` → `http://localhost:5174/tools/benchmark/?label=<laptop>` →
**Salin sebagai Markdown** → tempel di sini. Sertakan juga *Transferred* dari DevTools (bagian 2).

## 4. HP — diisi pemilik proyek

Langkah: `tools/benchmark/README.md` bagian "HP". Sebutkan model HP & versi Chrome di `label`.
