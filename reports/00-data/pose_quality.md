# Kualitas pose per kelas

Perintah: `python -m repcount.evaluation.pose_quality` · model `pose_landmarker_lite` (sama dengan web app), maks 30 fps, sisi terpanjang 640px.

651 video dengan keypoint (1 video tanpa keypoint, lihat `errors.csv`). Visibility = rata-rata kiri+kanan pada frame dengan pose terdeteksi. Diurutkan dari deteksi terburuk.

| Kelas | Video | Durasi (menit) | Pose terdeteksi | bahu | siku | pergelangan tangan | pinggul | lutut | pergelangan kaki | Sisi dominan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decline_bench_press | 11 | 2.6 | 87.6% | 1.00 | 0.70 | 0.68 | 0.97 | 0.60 | 0.46 | kanan (45%) |
| romanian_deadlift | 14 | 4.3 | 88.3% | 1.00 | 0.58 | 0.55 | 0.98 | 0.59 | 0.59 | kiri (57%) |
| bench_press | 61 | 3.8 | 91.7% | 1.00 | 0.65 | 0.65 | 0.96 | 0.59 | 0.45 | kiri (44%) |
| lat_pulldown | 51 | 3.7 | 94.8% | 0.99 | 0.76 | 0.75 | 0.85 | 0.41 | 0.27 | kanan (49%) |
| leg_extension | 25 | 3.4 | 95.1% | 1.00 | 0.76 | 0.71 | 0.99 | 0.81 | 0.75 | kiri (62%) |
| incline_bench_press | 33 | 4.4 | 97.6% | 0.99 | 0.74 | 0.71 | 0.68 | 0.42 | 0.24 | kanan (42%) |
| barbell_biceps_curl | 62 | 4.0 | 97.8% | 1.00 | 0.82 | 0.79 | 0.89 | 0.41 | 0.29 | seimbang (56%) |
| tricep_pushdown | 50 | 3.3 | 97.9% | 1.00 | 0.64 | 0.63 | 0.94 | 0.45 | 0.27 | kiri (50%) |
| deadlift | 32 | 3.1 | 98.7% | 1.00 | 0.75 | 0.73 | 1.00 | 0.79 | 0.76 | seimbang (47%) |
| hammer_curl | 19 | 3.1 | 98.9% | 1.00 | 0.79 | 0.80 | 0.95 | 0.56 | 0.35 | kiri (37%) |
| tricep_dips | 20 | 4.2 | 99.0% | 1.00 | 0.68 | 0.62 | 0.98 | 0.56 | 0.37 | kiri (55%) |
| hip_thrust | 18 | 3.5 | 99.1% | 1.00 | 0.67 | 0.68 | 0.98 | 0.80 | 0.75 | kanan (50%) |
| russian_twist | 13 | 3.6 | 99.2% | 1.00 | 0.72 | 0.66 | 0.96 | 0.73 | 0.66 | kanan (54%) |
| plank | 7 | 5.6 | 99.5% | 1.00 | 0.69 | 0.64 | 1.00 | 0.60 | 0.66 | kanan (86%) |
| push_up | 56 | 5.5 | 99.6% | 1.00 | 0.76 | 0.78 | 0.99 | 0.47 | 0.49 | kiri (52%) |
| pull_up | 26 | 3.5 | 99.7% | 1.00 | 0.92 | 0.90 | 0.93 | 0.63 | 0.58 | seimbang (73%) |
| squat | 29 | 5.0 | 99.8% | 1.00 | 0.75 | 0.71 | 0.97 | 0.70 | 0.59 | kiri (41%) |
| lateral_raise | 37 | 5.1 | 99.8% | 1.00 | 0.91 | 0.87 | 0.92 | 0.48 | 0.24 | seimbang (46%) |
| leg_raises | 21 | 3.2 | 99.9% | 1.00 | 0.80 | 0.80 | 0.99 | 0.71 | 0.63 | kanan (71%) |
| chest_fly_machine | 28 | 2.9 | 99.9% | 1.00 | 0.88 | 0.89 | 0.90 | 0.68 | 0.51 | seimbang (50%) |
| t_bar_row | 21 | 4.2 | 100.0% | 1.00 | 0.77 | 0.73 | 0.99 | 0.67 | 0.57 | kiri (43%) |
| shoulder_press | 17 | 3.0 | 100.0% | 1.00 | 0.96 | 0.96 | 0.79 | 0.39 | 0.21 | seimbang (76%) |

## Temuan

1. Pose terdeteksi pada 97.7% dari seluruh frame. Terburuk: **decline_bench_press** (87.6%), terbaik: **shoulder_press** (100.0%).
2. Kelas dengan deteksi < 80%: tidak ada; namun 8 video punya deteksi < 50% (bench_press 4, lat_pulldown 2, leg_extension 2).
3. Kelompok sendi dengan visibility rata-rata terendah: **pergelangan kaki** (0.49); paling rendah pada shoulder_press, lateral_raise, incline_bench_press.
4. Sisi tubuh dominan per video: kiri 38%, kanan 30%, seimbang 32% (pemilihan sisi otomatis di app tetap diperlukan).
5. Video tanpa satu pun pose terdeteksi: 3 (lat_pulldown_25, lat_pulldown_6, leg_extension_14).
