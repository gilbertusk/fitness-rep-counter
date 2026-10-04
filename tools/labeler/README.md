# tools/labeler — alat labeling repetisi

Web statis tanpa build step. Video dibuka langsung dari disk lewat pemilih folder; tidak ada yang
diunggah ke mana pun. Definisi satu rep dan aturan labeling: [`labels/README.md`](../../labels/README.md).

```bash
npx serve tools/labeler          # dari root repo, lalu buka alamat yang dicetak di Chrome atau Edge
```

1. **Muat to_label.csv** → `labels/to_label.csv` (dibuat oleh `python -m repcount.labels.select`).
2. **Pilih folder video** → `data/workout-videos/`. Status menunjukkan berapa video yang cocok;
   pencocokan memakai aturan `video_id` yang sama dengan manifest (`push-up_17.mp4` → `push_up_17`).
3. Isi **Nama pelabel**, lalu kerjakan dengan keyboard (daftar tombol ada di panel kanan alat):
   `Spasi` putar/jeda · `←/→` per frame · `,`/`.` kecepatan · `R` tandai rep · `Backspace` hapus tanda
   terakhir · `S`/`E` plank · `F` ambigu · `N` video berikutnya.
4. **Export CSV** → simpan sebagai `labels/rep_labels.csv` → `python -m repcount.labels.validate`.

Hanya label yang selesai (atau ambigu) yang diekspor. Masalah yang akan ditolak validator — misalnya
ambigu tanpa catatan — ditampilkan merah di bawah video sebelum export.

**Mode cek ulang** (minimal 2 hari setelah sesi utama): memilih ±10% video secara deterministik dari
`video_id`, menyimpan label di tempat terpisah sehingga hasil sesi utama tidak terlihat, dan
mengekspor `rep_labels_recheck.csv`. Lalu: `python -m repcount.labels.validate --agreement`.

## Penyimpanan

Label tersimpan otomatis di `localStorage` browser ini, per sesi. Muat ulang halaman aman: daftar
video, label, dan nama pelabel kembali — hanya folder video yang perlu dipilih lagi (browser tidak
mengizinkan akses ulang ke berkas lokal). Penyimpanan ini **hanya di satu browser, satu mesin**, dan
hilang bila data situs dihapus, jadi **Export berkala**. Untuk melanjutkan di mesin lain: **Impor label
CSV** dari hasil export sebelumnya.

## Video tidak bisa diputar

`.MOV` dari ponsel sering memakai codec HEVC (H.265), yang tidak selalu didukung browser. Coba Chrome
atau Edge terbaru; bila tetap gagal, konversi dan simpan di folder yang sama:

```bash
ffmpeg -i nama.MOV -c:v libx264 -crf 18 nama.mp4
```

Nama dasar berkas jangan diubah. Bila kedua berkas ada, alat otomatis memilih `.mp4`.

## Kode

- `src/labels.js` — logika murni (CSV, state label, aturan); diuji oleh `tests/labels.test.js` lewat `npm test`
- `src/app.js` — DOM, keyboard, dan `localStorage` saja
