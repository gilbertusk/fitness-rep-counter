# Tahap 5 — Siap Dipamerkan

> Salin seluruh isi file ini sebagai prompt ke AI.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md`** (§4 struktur folder — **wajib diikuti**) dan semua isi `reports/`. Tahap 0–4 sudah selesai. Kerjakan **hanya Tahap 5**.

## Tujuan

Membuat proyek ini meyakinkan bagi recruiter/engineer dalam 60 detik pertama: demo live, angka jelas, struktur rapi, hasil bisa direproduksi.

## Lokasi file tahap ini

```
tools/benchmark/               index.html, src/, README.md
reports/05-performance/        performance.md
.github/workflows/ci.yml
docs/MODEL_CARD.md
README.md (root)
```

## Tugas

1. **Benchmark** — `tools/benchmark/` → `reports/05-performance/performance.md`
   - Ukur di browser: FPS end-to-end, latensi per tahap (pose, fitur, klasifikasi, counter) median & p95, ukuran aset, waktu muat pertama.
   - Laptop (Chrome) oleh kamu; HP oleh saya — siapkan instruksinya.

2. **CI** — `.github/workflows/ci.yml`
   - JS: `npm test` + coverage (gagal jika < 80% untuk `app/src/core/`), ESLint config minimal.
   - Python: `pip install -e "ml[dev]"`, `pytest ml/tests --cov=repcount`, `ruff check ml`, termasuk parity test fitur.
   - **Cek struktur**: skrip kecil `tools/checkStructure.js` yang gagal jika ada file di root di luar daftar §4, atau jika `app/src/core/` meng-import `adapters/`/`ui/`.

3. **Deploy** — GitHub Pages atau Vercel, **hanya folder `app/`**. Pastikan model ONNX & MediaPipe termuat dari path produksi. Siapkan konfigurasi; deploy setelah saya setujui.

4. **README root final** (bahasa Indonesia, ringkasan Inggris di atas), urutan:
   1. Judul, satu kalimat, demo link + GIF (≤ 5 MB).
   2. Tabel hasil: klasifikasi (macro-F1, akurasi video), repetisi (MAE, OBO vs baseline), performa (FPS laptop/HP) — **setiap angka link ke file di `reports/`**.
   3. Diagram arsitektur.
   4. **Struktur repo** (pohon ringkas dari §4, 1 baris penjelasan per folder).
   5. Keputusan teknis penting (dari Log Keputusan).
   6. Keterbatasan & kegagalan yang diketahui.
   7. Cara reproduksi (dataset → laporan) dalam urutan perintah.
   8. Kredit dataset, lisensi, catatan privasi.

5. **Model card** — `docs/MODEL_CARD.md`: data latih, kelas, metrik, batasan (sudut kamera, 1 orang, pencahayaan), penggunaan yang tidak disarankan (bukan alat medis).

6. **Bersih-bersih**
   - Hapus kode mati, `console.log`, TODO basi; file > 400 baris dipecah.
   - Semua `README.md` per folder konsisten dan up to date.
   - `git ls-files` tidak berisi video (kecuali fixture kecil), `.npz`, atau rahasia.

## Kriteria selesai

- [ ] CI hijau, termasuk cek struktur.
- [ ] `reports/05-performance/performance.md` terisi (bagian laptop); instruksi HP siap.
- [ ] README root & MODEL_CARD final; semua angka punya sumber di `reports/`.
- [ ] Konfigurasi deploy siap, menunggu persetujuan.
- [ ] `docs/PLAN.md`: semua status ✅, Log Keputusan lengkap.

## Laporan akhir

Checklist di atas, hal yang perlu saya isi (demo link, GIF, benchmark HP), dan 3 saran pengembangan lanjutan (mis. 3D pose lifting untuk kamera frontal, label form dari Fitness-AQA, VBT). Lalu berhenti.
