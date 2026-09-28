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

## Struktur

```
src/
  core/            # logika murni, tanpa DOM, dan semuanya di-unit-test
    angles.js      # perhitungan sudut
    exercises.js   # konfigurasi latihan & threshold
    pose.js        # landmark → sudut + peringatan form
    repCounter.js  # state machine penghitung (immutable)
  config.js        # URL model & library
  main.js          # UI: kamera/video, render, loop
tests/
```

## Roadmap

- [x] MVP: squat & push-up, penghitung + 2 aturan form, unit test
- [ ] Rekam 20–30 video uji (form benar/salah), lalu tuning threshold
- [ ] Evaluasi di [RepCount](https://svip-lab.github.io/dataset/RepCount_dataset.html): MAE jumlah repetisi, dicantumkan di README
- [ ] Tambah latihan: lunge, bicep curl, shoulder press
- [ ] Smoothing sudut (EMA) & indikator kepercayaan pose
- [ ] Umpan balik suara (Web Speech API)
- [ ] Deploy (Vercel / GitHub Pages) + GitHub Actions menjalankan `npm test`
- [ ] (Lanjutan) Klasifikasi jenis latihan otomatis dari urutan keypoint

## Keterbatasan

- Threshold masih nilai awal dan perlu dituning dengan data nyata.
- Kamera sebaiknya menghadap tubuh dari samping; sudut kamera frontal membuat sudut sendi kurang akurat.
- Hanya satu orang per frame.
