# Tahap 4 — Integrasi Web App

> Salin seluruh isi file ini sebagai prompt ke AI.

---

Kamu bekerja di repo `D:\Computer Vision\fitness-rep-counter`. **Baca `docs/PLAN.md`** (§3, §4 struktur folder — **wajib diikuti**, §6, §8), `reports/01-classifier/classifier.md`, dan `reports/03-rep-counter/rep_counter.md`. Tahap 0–3 sudah selesai. Kerjakan **hanya Tahap 4**.

## Tujuan

Web app yang mengenali latihan otomatis, menghitung repetisi dengan counter generik, memberi koreksi form untuk latihan yang aturannya jelas, dan punya mode plank — tetap 100% di browser.

## Lokasi file tahap ini

```
app/src/core/session/          session.js, speechQueue.js (rate limit ucapan, murni)
app/src/core/form/             measure.js, debounce.js
app/src/core/form/rules/       squat.js, pushUp.js, barbellBicepsCurl.js, lateralRaise.js, deadlift.js, shoulderPress.js, plank.js, index.js
app/src/core/counting/         holdTimer.js
app/src/core/geometry/         oneEuroFilter.js
app/src/adapters/              speech.js, storage.js
app/src/ui/                    camera.js, overlay.js, panel.js, history.js
app/tests/unit/…               mencerminkan src/core/
app/tests/e2e/                 smoke.spec.js
docs/FORM_RULES.md
```

`main.js` hanya merangkai: adapters → core → ui. Tidak ada logika perhitungan di `ui/` atau `adapters/`.

## Tugas

1. **Sesi** — `core/session/session.js`, state machine murni:
   `IDLE → DETECTING (prediksi stabil ≥ 2 window di atas ambang) → COUNTING → RESTING (diam > 3 detik) → DETECTING`.
   Label terkunci selama COUNTING; override manual (dropdown) menonaktifkan auto-detect.

2. **Aturan form** — `core/form/rules/`, satu file per latihan, format deklaratif (titik, jenis ukuran, ambang, pesan), didaftarkan di `rules/index.js`.
   - Pertahankan aturan squat & push-up lama.
   - Tambahkan hanya aturan yang jelas & terukur dari 2D: biceps curl (siku tidak maju), lateral raise (tangan tidak melewati bahu), deadlift (punggung tidak membungkuk), shoulder press (lockout penuh).
   - `docs/FORM_RULES.md`: sumber/alasan dan batasan tiap aturan.
   - `core/form/debounce.js`: peringatan harus bertahan ≥ 0,5 detik sebelum muncul.

3. **Mode plank** — `core/counting/holdTimer.js`: timer berjalan saat garis bahu–pinggul–pergelangan kaki dalam toleransi; laporkan waktu tahan terbaik.

4. **Smoothing & kualitas pose** — One Euro filter pada landmark sebelum fitur; indikator kualitas pose di UI ("Posisikan kamera dari samping", "Tubuh tidak terlihat penuh").

5. **Suara** — logika rate limit (maks 1 ucapan / 2 detik) murni di `core/session/speechQueue.js`, pemanggilan Web Speech API di `adapters/speech.js`; bisa dimatikan.

6. **Ringkasan & riwayat** — daftar set (latihan, rep, rep dengan peringatan, durasi); penyimpanan via `adapters/storage.js` (`localStorage` dalam try/catch); tombol hapus riwayat.

7. **UI** — latihan terdeteksi + confidence, jumlah rep, tahap sesi, peringatan, FPS; label "eksperimental" untuk latihan yang lemah di Tahap 3; responsif di HP (pilih kamera depan/belakang); bahasa Indonesia.

## Test

- node:test untuk session (termasuk override & RESTING), holdTimer, setiap aturan form dengan pose sintetis, debounce, rate limit suara.
- Playwright `app/tests/e2e/smoke.spec.js`: memutar `app/tests/fixtures/videos/push-up_17.mp4`, memastikan hitungan > 0 dan tidak ada error console. Tambahkan script `test:e2e` di `package.json`.

## Kriteria selesai

- [ ] Semua file di lokasi yang tercantum; `app/src/core/` tidak meng-import `adapters/` atau `ui/` (buktikan dengan grep di laporan).
- [ ] `npm test` lulus, coverage ≥ 80% untuk `app/src/core/`; `npm run test:e2e` lulus.
- [ ] Tidak ada request jaringan yang mengirim frame/video (diverifikasi, ditulis di README).
- [ ] Demo `npm start` berjalan untuk webcam, file video, dan ≥ 5 latihan.
- [ ] `app/README.md` dan `docs/PLAN.md` diperbarui.

## Laporan akhir

Fitur yang jalan, latihan didukung penuh vs eksperimental, masalah UX saat uji manual. Lalu berhenti.
