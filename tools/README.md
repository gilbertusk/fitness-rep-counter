# tools/ — Alat bantu developer (tidak di-deploy)

- `labeler/` — alat labeling repetisi (Tahap 2): `npx serve tools/labeler` — petunjuk di `labeler/README.md`
- `eval/` — evaluasi penghitung repetisi di Node (Tahap 3): `node tools/eval/evalReps.js` — petunjuk di `eval/README.md`
- `benchmark/` — FPS & latensi pipeline app di browser (Tahap 5) — petunjuk di `benchmark/README.md`
- `checkStructure.js` — penjaga struktur untuk CI (Tahap 5): `node tools/checkStructure.js`

Test `tools/**/*.test.js` ikut dijalankan oleh `npm test`.
