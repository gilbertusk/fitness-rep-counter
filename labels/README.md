# labels/ — Label buatan manusia (di-commit)

Ground truth untuk mengevaluasi penghitung repetisi (Tahap 3). **Label hanya dari manusia — tidak
pernah dibuat otomatis, tidak pernah ditebak dari model** (`docs/PLAN.md` §6.4).

| Berkas | Isi | Dibuat oleh |
|---|---|---|
| `to_label.csv` | daftar video yang perlu dilabel (±5 per kelas, dari split val + test) | `python -m repcount.labels.select` |
| `rep_labels.csv` | label utama | manusia, lewat `tools/labeler/` → Export CSV |
| `rep_labels_recheck.csv` | label ulang ±10% video, beberapa hari kemudian (untuk mengukur konsistensi) | manusia, mode **cek ulang** di alat yang sama |

Cek setelah melabel: `python -m repcount.labels.validate` (tambahkan `--agreement` setelah sesi cek ulang).

## Satu repetisi

**Aturan umum: satu rep = satu siklus penuh, dan tandanya diletakkan saat gerakan *kembali ke posisi
awal*.** Bukan di titik terdalam, bukan di tengah. Penghitung di Tahap 3 juga menambah hitungan saat
siklus selesai, jadi konvensi yang sama membuat waktu tanda bisa dibandingkan.

| Kelas | Posisi awal | Tekan `R` saat… |
|---|---|---|
| barbell_biceps_curl | lengan lurus ke bawah | lengan kembali lurus setelah mengangkat |
| hammer_curl | lengan lurus ke bawah, genggaman netral | lengan kembali lurus — **bergantian: tiap lengan = 1 rep** |
| bench_press | lengan lurus, beban di atas dada | lengan kembali lurus setelah beban turun ke dada |
| incline_bench_press | sama dengan bench press | lengan kembali lurus |
| decline_bench_press | sama dengan bench press | lengan kembali lurus |
| chest_fly_machine | lengan terbuka ke samping | lengan kembali terbuka setelah bertemu di depan |
| deadlift | beban di lantai | beban kembali menyentuh lantai / titik terendah |
| romanian_deadlift | berdiri tegak, beban di depan paha | kembali berdiri tegak setelah membungkuk |
| hip_thrust | pinggul di bawah | pinggul kembali turun setelah diangkat |
| lat_pulldown | lengan lurus ke atas | lengan kembali lurus setelah bar ditarik ke dada |
| t_bar_row | lengan lurus, beban menggantung | lengan kembali lurus setelah menarik |
| pull_up | menggantung, lengan lurus | kembali menggantung setelah dagu melewati bar |
| push_up | lengan lurus (posisi atas) | lengan kembali lurus setelah dada turun |
| tricep_dips | lengan lurus (posisi atas) | lengan kembali lurus setelah badan turun |
| tricep_pushdown | siku tertekuk, pegangan di depan dada | pegangan kembali naik setelah lengan lurus ke bawah |
| shoulder_press | beban di samping bahu | beban kembali ke bahu setelah didorong ke atas |
| lateral_raise | lengan di samping badan | lengan kembali turun setelah naik setinggi bahu |
| leg_extension | lutut tertekuk | lutut kembali tertekuk setelah kaki lurus |
| leg_raises | kaki di bawah (berbaring atau menggantung) | kaki kembali turun setelah diangkat |
| squat | berdiri tegak | kembali berdiri tegak setelah turun |
| russian_twist | tangan di satu sisi | tangan kembali ke **sisi yang sama** dengan sentuhan pertama (kiri + kanan = 1 rep) |
| **plank** | — | **bukan repetisi**: `S` saat posisi plank tercapai, `E` saat posisi lepas (lutut turun / pinggul jatuh jelas) |

Gerakan dua lengan yang dilakukan **bergantian** (hammer curl, kadang shoulder press atau lateral
raise): tiap lengan = 1 rep, dan tulis `bergantian` di catatan. Tahap 3 akan menganalisisnya terpisah.

## Kasus yang perlu aturan

- **Rep parsial** (rentang gerak jelas tidak tercapai, mis. squat hanya seperempat turun): **tidak
  dihitung**. Kalau ragu apakah cukup dalam → hitung, lalu tandai video ambigu (`F`) dan jelaskan.
- **Video mulai di tengah rep**: hitung hanya jika fase utama (mengangkat / menurunkan) terlihat utuh.
- **Video habis sebelum rep kembali ke posisi awal**: hitung jika fase utama selesai; tandai di frame
  terakhir dan tulis `terpotong` di catatan.
- **Plank yang sudah berlangsung saat video mulai**: `S` di 0 ms, tulis `mulai sebelum video`. Plank yang
  belum lepas saat video habis: `E` di frame terakhir, tulis `berlanjut setelah video`.
- **Bukan rep**: pemanasan, mengatur pegangan, mengangkat bar dari rak atau mengembalikannya.
- **Beberapa set dengan jeda**: hitung semua rep dari semua set.
- **Lebih dari satu orang**: label orang utama (paling besar / di tengah), tandai ambigu.
- **Video tidak bisa diputar atau isinya bukan latihan itu**: tandai ambigu dengan alasannya, jangan
  dikarang.

Ambigu (`F`) berlaku untuk **seluruh video**, dan wajib disertai catatan. Video ambigu tetap disimpan;
Tahap 3 melaporkan hasil dengan dan tanpa video ambigu.

## Format `rep_labels.csv`

```
video_id,label,rep_count,rep_timestamps_ms,hold_start_ms,hold_end_ms,is_ambiguous,notes,labeler,labeled_at
```

- `rep_timestamps_ms`: milidetik dari awal video, naik, dipisah `;`. `rep_count` = jumlahnya.
- Plank: `rep_count` dan `rep_timestamps_ms` kosong; `hold_start_ms` < `hold_end_ms`.
- `labeler`: nama/inisial pemberi label. `labeled_at`: waktu ISO 8601, diisi otomatis oleh alat.

## Panduan kerja

1. Di mesin yang punya dataset: `python -m repcount.labels.select` → `labels/to_label.csv`.
2. `npx serve tools/labeler`, buka alamatnya di **Chrome atau Edge** (petunjuk lengkap:
   `tools/labeler/README.md`). Muat `to_label.csv`, lalu pilih folder `data/workout-videos/`.
3. **Kerjakan per kelas**, berurutan — definisi gerakan tetap segar di kepala, hasil lebih konsisten.
4. Kelas cepat (curl, bench press, pushdown — rata-rata < 4 detik per video): pakai kecepatan 0,5×.
5. Selesai satu sesi: **Export CSV** → simpan sebagai `labels/rep_labels.csv` → jalankan validasi.
6. **Minimal 2 hari kemudian**: aktifkan mode **cek ulang** di alat (memilih ±10% video secara
   deterministik dan menyembunyikan label lama), label tanpa melihat hasil sebelumnya, Export →
   `labels/rep_labels_recheck.csv` → `python -m repcount.labels.validate --agreement`.

**Perkiraan waktu: ±2 jam** untuk ±105 video, ditambah ±15 menit untuk cek ulang. Dasarnya: rata-rata
video di dataset ini 7,8 detik (`reports/00-data/pose_quality.csv`), jadi rekaman mentahnya hanya
±14 menit; sisanya habis untuk memutar pelan, maju per frame, dan memeriksa ulang. Kelas panjang
(plank 48 detik, romanian deadlift 18 detik, russian twist 17 detik) memakan waktu jauh lebih banyak
per video daripada bench press (4 detik). Istirahat tiap ±30 menit — kelelahan menurunkan konsistensi.
