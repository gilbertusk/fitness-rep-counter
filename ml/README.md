# ml/ — Pipeline Python (`repcount`)

Ekstraksi keypoint, split data, dan (tahap berikutnya) pelatihan & evaluasi model.
Semua data dibaca/ditulis di `data/` root repo (override: env `FITNESS_DATA_DIR`).

```bash
python -m venv .venv && .venv/Scripts/activate      # dari root repo
pip install -e "ml[dev]"
pytest ml/tests --cov=repcount && ruff check ml
python -m repcount.data.extract --limit 5           # lihat README root: "Pipeline data"
```

- `src/repcount/` — kode paket; `tests/` — mencerminkan struktur `src/repcount/`
- `splits/` — split train/val/test (di-commit)
