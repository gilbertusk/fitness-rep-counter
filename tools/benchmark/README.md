# tools/benchmark/ — FPS & latensi di perangkat nyata

Mengukur pipeline app yang **sebenarnya** (modul `app/src/` di-import langsung, bukan salinan):

| Ukuran | Cara |
|---|---|
| FPS end-to-end | pose → `stepFrame` → gambar overlay, pada kecepatan video asli (maks. FPS video) |
| Pose, pengenal | diukur per panggilan saat jalan langsung |
| Fitur, penghitung, core total | landmark yang terekam diputar ulang; tiap panggilan diulang 20× pada input yang sama lalu dirata-rata (fungsi core murni, dan di bawah resolusi 0,1 ms `performance.now()`) |
| Ukuran aset, waktu muat | `PerformanceResourceTiming`; waktu dihitung dari halaman dibuka |

Hasilnya median, p95, dan maks per tahap, lalu tombol **Salin sebagai Markdown** → tempel di
`reports/05-performance/performance.md`. Video tidak dikirim ke mana pun.

## Laptop (Chrome/Edge)

```bash
npx --yes serve -l 5174 .        # dari ROOT repo (halaman meng-import ../../app/src)
```

Buka `http://localhost:5174/tools/benchmark/?label=Laptop%20X`, tutup tab lain, colokkan charger,
lalu **Mulai benchmark**. Opsi URL: `repeat=3` (putaran video), `exercise=push_up`,
`model=…&labels=…` (default `app/models/`).

## HP (yang harus diukur pemilik proyek)

Benchmark memakai video fixture, bukan kamera, jadi tidak butuh HTTPS. Halaman ini tidak ikut
di-deploy (hanya `app/`), jadi HP membukanya dari laptop:

1. Laptop & HP di Wi-Fi yang sama; di laptop jalankan `npx --yes serve -l 5174 .` dari root repo.
2. Di HP buka `http://<IP-laptop>:5174/tools/benchmark/?label=<model HP>`.
   (Tombol salin butuh konteks aman; bila gagal, salin teks laporan secara manual — atau pakai
   `chrome://inspect` → *Port forwarding* `5174 → localhost:5174` lewat USB dan buka `http://localhost:5174/...`.)
3. Layar menyala, mode hemat daya mati, baterai > 50%, lalu **Mulai benchmark**.
4. Tempel hasilnya di `reports/05-performance/performance.md` bagian "HP".

## Headless (CI / tanpa layar)

```bash
node tools/benchmark/headless.js --repeat 5 [--model x.onnx --labels labels.json]
```

Chromium headless tanpa GPU: mengukur jalur CPU. Aset diambil dari `node_modules/` / `data/` bila
ada (tanpa kompresi HTTP, jadi kolom "unduh" = ukuran file asli). Lihat komentar di `headless.js`.

`src/stats.js` murni (diuji di `tests/stats.test.js`); `src/bench.js` = pengukuran di browser.
