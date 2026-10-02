# tools/ — Alat bantu developer (tidak di-deploy)

- `labeler/` — alat labeling repetisi (Tahap 2): `npx serve tools/labeler` — petunjuk di `labeler/README.md`

Direncanakan (lihat `docs/PLAN.md` §4):

- `eval/` — evaluasi penghitung repetisi di Node (Tahap 3)
- `benchmark/` — benchmark FPS di browser (Tahap 5)
- `checkStructure.js` — penjaga struktur folder untuk CI (Tahap 5)

Test `tools/**/*.test.js` ikut dijalankan oleh `npm test`.
