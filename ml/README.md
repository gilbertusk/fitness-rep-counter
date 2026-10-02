# ml/ — Pipeline Python (`repcount`)

Ekstraksi keypoint, split data, pelatihan & evaluasi pengenal jenis latihan.
Semua data dibaca/ditulis di `data/` root repo (override: env `FITNESS_DATA_DIR`).

```bash
python -m venv .venv && .venv/Scripts/activate      # dari root repo
pip install -e "ml[dev,train]"                      # extra train: torch, sklearn, onnx
pytest ml/tests --cov=repcount && ruff check ml
python -m repcount.data.extract --limit 5           # Tahap 0 — lihat README root
```

Tahap 1 (butuh `data/keypoints/` dari Tahap 0); `RUN=data/runs/<timestamp>`:

```bash
python -m repcount.features.golden                  # fixture parity fitur (tanpa dataset)
python -m repcount.features.windows                 # cek jumlah window per split
python -m repcount.models.baseline    --out $RUN
python -m repcount.models.temporal    --out $RUN
python -m repcount.evaluation.classifier_report --run $RUN   # test set dipakai SEKALI di sini
python -m repcount.export.onnx        --run $RUN             # → app/models/
```

Tahap 2 (butuh `data/keypoints/manifest.csv`; labelnya dari manusia lewat `tools/labeler/`):

```bash
python -m repcount.labels.select                    # → labels/to_label.csv
python -m repcount.labels.validate                  # cek labels/rep_labels.csv
python -m repcount.labels.validate --agreement      # + konsistensi dengan rep_labels_recheck.csv
```

- `src/repcount/` — kode paket; `tests/` — mencerminkan struktur `src/repcount/`
- `splits/` — split train/val/test (di-commit)
- Spesifikasi fitur: `docs/FEATURES.md` — wajib sinkron dengan `app/src/core/features/features.js`
