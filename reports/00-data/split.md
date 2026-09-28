# Split data v1

Perintah: `python -m repcount.data.split` · seed 42 · rasio 70/15/15 per kelas, dibagi per `group_id`.

- Video dalam split: **651** (dikecualikan: 1 — tanpa keypoint: dbp_4)
- Grup near-duplicate berisi > 1 video: **70** (184 video); satu grup selalu berada di satu split.
- Total: train **453** (69.6%), val **99** (15.2%), test **99** (15.2%)

| Kelas | train | val | test | total |
|---|---:|---:|---:|---:|
| barbell_biceps_curl | 44 | 9 | 9 | 62 |
| bench_press | 43 | 9 | 9 | 61 |
| chest_fly_machine | 20 | 4 | 4 | 28 |
| deadlift | 22 | 5 | 5 | 32 |
| decline_bench_press | 7 | 2 | 2 | 11 |
| hammer_curl | 13 | 3 | 3 | 19 |
| hip_thrust | 12 | 3 | 3 | 18 |
| incline_bench_press | 23 | 5 | 5 | 33 |
| lat_pulldown | 35 | 8 | 8 | 51 |
| lateral_raise | 25 | 6 | 6 | 37 |
| leg_extension | 17 | 4 | 4 | 25 |
| leg_raises | 15 | 3 | 3 | 21 |
| plank | 5 | 1 | 1 | 7 |
| pull_up | 18 | 4 | 4 | 26 |
| push_up | 40 | 8 | 8 | 56 |
| romanian_deadlift | 10 | 2 | 2 | 14 |
| russian_twist | 9 | 2 | 2 | 13 |
| shoulder_press | 11 | 3 | 3 | 17 |
| squat | 21 | 4 | 4 | 29 |
| t_bar_row | 15 | 3 | 3 | 21 |
| tricep_dips | 14 | 3 | 3 | 20 |
| tricep_pushdown | 34 | 8 | 8 | 50 |

## Pengecualian

- plank: hanya 7 video (< 10); train/val/test = 5/1/1
