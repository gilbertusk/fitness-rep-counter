# Aturan Form

Aturan koreksi form yang ditampilkan app (`app/src/core/form/rules/`). Satu berkas per latihan,
formatnya deklaratif (titik tubuh, jenis ukuran, ambang, pesan), dievaluasi oleh
`evaluateForm()` di `app/src/core/form/measure.js`.

**Status: semua ambang di bawah adalah titik awal, belum divalidasi dengan data.** Squat dan push-up
dibawa apa adanya dari app awal; aturan lain disusun dari penalaran biomekanik yang umum dipakai dalam
panduan latihan kekuatan, bukan dari studi tertentu. Dataset proyek ini tidak punya label form
(`docs/PLAN.md` §2), jadi akurasi peringatan belum bisa diukur — itu pekerjaan lanjutan yang butuh
video form benar/salah berlabel.

## Prinsip

- **Hanya yang terukur dari satu kamera 2D.** Kedalaman (`z`) MediaPipe tidak dipakai karena tidak
  stabil (`docs/FEATURES.md` §0). Aturan yang butuh 3D (lutut masuk ke dalam saat squat, rotasi
  pinggul) sengaja tidak dibuat.
- **Sisi tubuh yang paling terlihat** dipakai, dengan visibility minimal 0,5 di titik-titik
  `sidePoints`; kalau tidak ada sisi yang memenuhi, tidak ada peringatan sama sekali — lebih baik
  diam daripada mengoreksi berdasarkan titik yang ditebak.
- **Debounce 0,5 detik** (`core/form/debounce.js`): peringatan baru muncul setelah aturan dilanggar
  terus-menerus selama itu, supaya satu frame tracking yang buruk tidak memunculkan koreksi.
- **Ukuran di ruang piksel** (x × lebar, y × tinggi), sehingga sudut tidak terdistorsi rasio aspek.
- **Landmark dihaluskan One Euro filter** sebelum diukur (`core/geometry/oneEuroFilter.js`).
  Classifier dan penghitung repetisi tetap memakai landmark mentah — lihat `docs/PLAN.md` §8.

## Aturan per latihan

| Latihan | Pandangan | Ukuran | Peringatan bila | Pesan |
|---|---|---|---|---|
| Squat | samping | sudut ruas bahu–pinggul dari vertikal | > 45° | Punggung terlalu membungkuk |
| Push-up | samping | sudut bahu–pinggul–pergelangan kaki | < 160° | Jaga badan tetap lurus |
| Barbell biceps curl | samping | sudut di bahu antara lengan atas dan badan | > 30° | Siku jangan maju — jaga lengan atas di samping badan |
| Lateral raise | bebas | tinggi pergelangan tangan di atas bahu, dalam panjang torso | > 0,15 | Jangan angkat tangan melewati bahu |
| Deadlift | samping | sudut telinga–bahu–pinggul | < 150° | Jaga punggung tetap lurus |
| Shoulder press | bebas | sudut siku **terbesar dalam satu rep** | < 160° | Luruskan lengan penuh di atas kepala |
| Plank | samping | jarak pinggul dari garis bahu–pergelangan kaki, dalam panjang garis (+ = di bawah) | > 0,10 atau < −0,12 | Pinggul turun — kencangkan perut / Pinggul terlalu tinggi |

### Alasan dan batasan

**Squat — punggung membungkuk.** Dari app awal. Membungkuk ke depan memang bagian normal dari squat,
apalagi bagi orang dengan tungkai panjang; 45° adalah batas kasar, bukan diagnosis. Tidak bisa
membedakan condong yang wajar (low-bar squat) dari punggung yang membulat.

**Push-up — badan lurus.** Dari app awal. Sudut di pinggul tidak membedakan pinggul turun dari
pinggul naik — keduanya mengecilkan sudut. Pesannya umum ("lurus") karena alasan itu.

**Biceps curl — siku maju.** Saat curl, lengan atas idealnya tetap di samping badan; siku yang
maju memindahkan beban ke bahu. Di puncak gerakan siku wajar maju sedikit, jadi ambangnya 30°, bukan
lebih ketat. Butuh pandangan samping: dari depan, siku yang maju tidak terlihat.

**Lateral raise — tidak melewati bahu.** Mengangkat tangan jauh di atas bahu umumnya dihindari
pada gerakan ini; targetnya kira-kira setinggi bahu. Toleransi 0,15 panjang torso memberi ruang untuk
tracking pergelangan tangan yang meleset. Terbaca dari depan maupun samping.

**Deadlift — punggung lurus (proksi).** MediaPipe tidak punya titik di tulang belakang, jadi yang
diukur adalah apakah telinga, bahu, dan pinggul tetap kira-kira segaris. Punggung atas yang membulat
mendorong bahu keluar dari garis itu. **Batasan besar:** kepala yang mendongak atau menunduk juga
menggeser telinga dan bisa memicu peringatan palsu; lengkungan punggung *bawah* (yang paling sering
dikhawatirkan) hampir tidak terlihat dari titik-titik ini. Anggap peringatan ini petunjuk kasar.

**Shoulder press — lockout penuh.** Dinilai sekali per rep, saat rep selesai, pada sudut siku
terlurus selama rep itu — bukan per frame, karena di separuh gerakan siku memang tertekuk. Sudut siku
terbaca dari depan maupun samping.

**Plank — garis badan.** Tinggi pinggul dibandingkan garis lurus bahu–pergelangan kaki, sehingga
pinggul turun dan pinggul naik bisa dibedakan (tidak seperti sudut push-up). Aturan ini juga yang
menentukan "dalam posisi" bagi timer plank (`core/counting/holdTimer.js`): timer hanya berjalan
selama kedua batas terpenuhi. Mengasumsikan badan kira-kira horizontal; untuk plank yang sangat miring
ukurannya kurang tepat.

## Menambah aturan

1. Buat `app/src/core/form/rules/<latihan>.js` dengan `view`, `sidePoints`, dan `rules`
   (`id`, `measure: { kind, points }`, `limit: { min | max }`, `message`, opsional `per: 'rep'`).
2. Daftarkan di `rules/index.js` dengan label dataset (snake_case).
3. Tambahkan test dengan pose sintetis di kedua sisi ambang (`app/tests/unit/form/rules/`).
4. Tambahkan barisnya di tabel dan alasannya di dokumen ini.

Jenis ukuran yang tersedia: `angle` (sudut di titik tengah), `angleFromVertical`, `heightAbove`,
`lineOffset`. Jenis baru ditambahkan di `MEASURES` dalam `measure.js`.
