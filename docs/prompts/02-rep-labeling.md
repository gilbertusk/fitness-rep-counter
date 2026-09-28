# Tahap 2 — Alat & Label Repetisi

> Salin seluruh isi file ini sebagai prompt ke AI. Tahap ini **butuh kerja manusia**: AI membuat alatnya, manusia yang memberi label.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md`** (§3, §4 struktur folder — **wajib diikuti**, §6). Tahap 0 sudah selesai (manifest & split tersedia). Kerjakan **hanya Tahap 2**.

## Tujuan

Label jumlah repetisi **beserta waktu setiap rep** untuk ±5 video per kelas (±110 video), sebagai ground truth evaluasi di Tahap 3. **Kamu tidak boleh membuat label sendiri** — hanya membangun alat dan daftar video.

## Lokasi file tahap ini

```
ml/src/repcount/labels/        select.py, validate.py
ml/tests/labels/
tools/labeler/                 index.html, styles.css, README.md
tools/labeler/src/             app.js (DOM), labels.js (logika murni: parse/validasi/serialisasi CSV)
tools/labeler/tests/           labels.test.js
labels/                        README.md, to_label.csv, rep_labels.csv
```

## Tugas

1. **Pilih video** — `python -m repcount.labels.select` → `labels/to_label.csv`
   - Dari split **val + test** saja; maks 5 video per kelas, utamakan variasi (ekstensi, durasi), seed tetap.
   - Plank dilabel sebagai durasi tahan (mulai & selesai), bukan repetisi.

2. **Alat labeling** — `tools/labeler/` (vanilla JS, tanpa build step, jalan dengan `npx serve tools/labeler`)
   - Membuka video lokal lewat file picker, memuat daftar dari `to_label.csv`.
   - Keyboard: `Spasi` play/pause, `←/→` mundur/maju ±1 frame, `,`/`.` kecepatan 0,5×/1×/2×, **`R` tandai satu rep**, `Backspace` hapus tanda terakhir, `N` video berikutnya, `F` tandai ambigu + catatan; plank: `S`/`E` mulai/selesai tahan.
   - Tampilkan jumlah tanda dan timeline; simpan otomatis ke `localStorage` (try/catch); tombol **Export CSV**.
   - Logika murni di `src/labels.js`, DOM di `src/app.js`.

3. **Format label** — `labels/rep_labels.csv`
   ```
   video_id,label,rep_count,rep_timestamps_ms,hold_start_ms,hold_end_ms,is_ambiguous,notes,labeler,labeled_at
   ```
   `rep_timestamps_ms` dipisah `;`. Di `labels/README.md`: definisi "satu repetisi" per jenis gerakan, aturan rep parsial, penanganan video terpotong, dan panduan kerja untuk manusia (urutan, perkiraan waktu).

4. **Validasi** — `python -m repcount.labels.validate`
   - `video_id` ada di manifest, `rep_count` = jumlah timestamp, timestamp naik & dalam durasi, tanpa duplikat.
   - Ringkasan per kelas: jumlah video berlabel, rata-rata & rentang rep.

5. **(Disarankan) Konsistensi label**: panduan melabel ulang 10% video acak beberapa hari kemudian, dan opsi `--agreement` di `validate` untuk menghitung selisih antar sesi.

## Kriteria selesai

- [ ] Semua file berada di lokasi yang tercantum; `tools/README.md` mencantumkan labeler.
- [ ] Alat labeling berjalan di browser; `npm test` (termasuk `tools/labeler/tests`) dan pytest lulus.
- [ ] `labels/to_label.csv` & `labels/README.md` di-commit.
- [ ] `docs/PLAN.md`: Status Tahap 2 = "alat siap, menunggu label manusia".

## Laporan akhir

Cara memakai alat, perkiraan waktu melabel, lalu **berhenti dan tunggu saya selesai melabel**. Setelah saya selesai, jalankan validasi dan laporkan hasilnya.
