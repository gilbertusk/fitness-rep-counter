# Spesifikasi Fitur (v1)

Dokumen ini adalah **kontrak** antara `ml/src/repcount/features/features.py` (Python, untuk training)
dan `app/src/core/features/features.js` (JS, untuk browser). Keduanya wajib menghasilkan angka yang
sama; dijaga oleh parity test terhadap `app/tests/fixtures/features_golden.json` (toleransi `1e-4`).

**Setiap perubahan di dokumen ini harus diikuti perubahan di kedua implementasi + regenerasi golden file.**

## 0. Masukan

Satu urutan pose mentah dari MediaPipe `pose_landmarker_lite`:

| Field | Bentuk | Keterangan |
|---|---|---|
| `landmarks` | `(T, 33, 4)` | `x`, `y` ternormalisasi ke `[0,1]` terhadap lebar/tinggi gambar; `z`; `visibility` |
| `timestamps_ms` | `(T,)` | stempel waktu tiap frame, milidetik, menaik |
| `width`, `height` | skalar | ukuran frame **sumber** (sebelum resize) |

Frame tanpa pose terdeteksi berisi `NaN` (lihat `repcount.data.extract.pose_to_arrays`).
`z` **tidak dipakai** — `pose_landmarker_lite` menghasilkan kedalaman yang tidak stabil untuk
latihan mesin dan posisi berbaring (lihat `reports/00-data/pose_quality.md`).

## 1. Koreksi rasio aspek

MediaPipe menormalisasi `x` terhadap lebar dan `y` terhadap tinggi, jadi pada frame non-persegi
satu satuan `x` ≠ satu satuan `y` dan sudut sendi menjadi terdistorsi. Sebelum apa pun:

```
aspect = width / height
x_iso  = x * aspect
y_iso  = y
```

Skala global tidak penting karena langkah §3 membaginya lagi.

## 2. Resample ke 15 fps

Sumber maksimal 30 fps (`config.MAX_FPS`), target **15 fps** (perioda 66,667 ms).

```
dt      = 1000 / 15
K       = floor((t[T-1] - t[0]) / dt) + 1
target_k = t[0] + k * dt            untuk k = 0 … K-1
```

Tiap `target_k` mengambil frame sumber dengan `|t[i] - target_k|` terkecil; seri sama → indeks terkecil.
Aturan ini **tidak menginterpolasi** posisi: frame asli dipakai apa adanya, jadi hasil di browser
(yang menerima frame satu per satu) identik dengan hasil offline.

Urutan lebih pendek dari satu window (< 30 frame setelah resample) tidak menghasilkan window.

## 3. Normalisasi per frame

Dihitung **per frame**, memakai koordinat hasil §1:

```
hip_mid    = mean(left_hip, right_hip)                      → pusat
torso_l    = |left_shoulder  - left_hip|                    (jarak Euclid)
torso_r    = |right_shoulder - right_hip|
scale      = mean(torso_l, torso_r)                         → skala
```

- `hip_mid` dan `scale` memakai rata-rata dari sisi yang **tidak NaN** saja, jadi satu pinggul atau
  satu bahu yang hilang masih menghasilkan frame yang sah.
- Bila keduanya NaN (`hip_mid` atau `scale` tak terhitung) → frame dianggap hilang.
- `scale < 1e-6` → frame dianggap hilang (pose merosot / semua titik menumpuk).
- Koordinat akhir: `(p - hip_mid) / scale`.

Hasilnya invarian terhadap posisi dan jarak orang ke kamera, tapi **tidak** terhadap rotasi kamera —
disengaja, karena orientasi tubuh membedakan bench press (berbaring) dari shoulder press (berdiri).

## 4. Landmark yang dipakai (13)

12 landmark tubuh utama + hidung, dengan urutan **tetap** (indeks di kolom kanan = indeks MediaPipe):

| # | Nama | MediaPipe |
|---:|---|---:|
| 0 | nose | 0 |
| 1 | left_shoulder | 11 |
| 2 | right_shoulder | 12 |
| 3 | left_elbow | 13 |
| 4 | right_elbow | 14 |
| 5 | left_wrist | 15 |
| 6 | right_wrist | 16 |
| 7 | left_hip | 23 |
| 8 | right_hip | 24 |
| 9 | left_knee | 25 |
| 10 | right_knee | 26 |
| 11 | left_ankle | 27 |
| 12 | right_ankle | 28 |

Landmark wajah selain hidung, jari, dan kaki bagian depan dibuang: tidak informatif untuk jenis
latihan dan menambah dimensi.

## 5. Sudut sendi (8)

Sudut di titik `b` antara ruas `b→a` dan `b→c`, dalam derajat, **dibagi 180** sehingga berada di `[0,1]`:

| # | Sudut | a | b | c |
|---:|---|---|---|---|
| 0 | left_elbow | left_shoulder | left_elbow | left_wrist |
| 1 | right_elbow | right_shoulder | right_elbow | right_wrist |
| 2 | left_shoulder | left_elbow | left_shoulder | left_hip |
| 3 | right_shoulder | right_elbow | right_shoulder | right_hip |
| 4 | left_hip | left_shoulder | left_hip | left_knee |
| 5 | right_hip | right_shoulder | right_hip | right_knee |
| 6 | left_knee | left_hip | left_knee | left_ankle |
| 7 | right_knee | right_hip | right_knee | right_ankle |

Titik hilang atau dua titik berimpit → sudut `NaN`. Sudut dihitung pada koordinat **ter-normalisasi**
(§3); karena normalisasi hanya translasi + skala seragam, nilainya sama dengan pada koordinat `_iso`.

## 6. Vektor fitur per frame (47)

Urutan **tetap**, inilah yang diperbandingkan parity test:

```
[0  … 38]  13 landmark × (x, y, visibility)   → 39 nilai, urutan §4
[39 … 46]  8 sudut sendi                      → 8 nilai,  urutan §5
```

`visibility` diambil apa adanya dari MediaPipe (`[0,1]`), tidak dinormalisasi.
Frame yang dianggap hilang (§3) berisi `NaN` pada **seluruh** 47 nilai.

## 7. Penanganan NaN

Dilakukan **per kolom**, pada urutan hasil resample, sebelum dipotong menjadi window:

1. **Gap ≤ 5 frame** (≈ 0,33 detik) di antara dua nilai valid → interpolasi linear.
2. **Gap > 5 frame** → dibiarkan `NaN`.
3. `NaN` di awal/akhir urutan (tanpa nilai valid di salah satu sisi) → **tidak** diekstrapolasi.

Frame disebut **hilang** bila `NaN` masih tersisa di kolom mana pun setelah langkah di atas.

## 8. Window

- Panjang **30 frame** (2 detik pada 15 fps), stride **15 frame** (tumpang tindih 50%).
- Window dengan **> 30% frame hilang** (yaitu > 9 dari 30):
  - **training / evaluasi**: dibuang;
  - **inferensi**: tetap dihitung tetapi ditandai `uncertain = true`.
- `NaN` yang tersisa di window yang dipakai diisi **0** (= posisi pinggul, sudut 0°) agar model
  menerima angka; `visibility` yang menyertainya sudah rendah sehingga model bisa belajar mengabaikannya.

Bentuk keluaran satu window: `(30, 47)`.

## 9. Augmentasi flip kiri–kanan (hanya saat training)

1. Tukar pasangan landmark kiri↔kanan (§4: 1↔2, 3↔4, 5↔6, 7↔8, 9↔10, 11↔12; `nose` tetap).
2. Negasikan `x` yang **sudah ter-normalisasi** (bukan `x_iso`).
3. Tukar pasangan sudut kiri↔kanan (§5: 0↔1, 2↔3, 4↔5, 6↔7). Nilai sudut tidak berubah tandanya.

Label tidak berubah: tidak ada kelas di dataset yang bergantung pada sisi tubuh.

## 10. Reproduksi

```bash
python -m repcount.features.golden        # tulis ulang app/tests/fixtures/features_golden.json
pytest ml/tests/features -q               # parity sisi Python
npm test                                  # parity sisi JS
```

Golden file dibuat dari **urutan pose sintetis deterministik** (seed 42), bukan dari video dataset,
supaya parity test bisa dijalankan tanpa mengunduh 4,6 GB data dan tetap reproducible di CI.
Urutan itu sengaja memuat kasus tepi: gap NaN pendek (diinterpolasi), gap NaN panjang (dibiarkan),
visibility rendah, dan frame dengan skala merosot. Untuk memeriksa dengan video nyata:
`python -m repcount.features.golden --video-id push_up_17`.
