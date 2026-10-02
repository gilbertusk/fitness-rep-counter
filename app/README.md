# app/ — Web app (yang di-deploy)

MediaPipe Pose → penghitung repetisi generik + aturan form + timer plank + pengenal latihan (bila model ada). 100% di browser.
Di-deploy apa adanya, tanpa `tests/` dan README ini, oleh `.github/workflows/deploy-pages.yml` (manual).

- `src/core/` — logika murni tanpa DOM, 100% baris ter-test; `session/workout.js` = seluruh keputusan per frame
- `src/adapters/` — browser API (MediaPipe, ONNX, suara, localStorage); `src/ui/` — render DOM & kanvas
- `src/main.js` — hanya merangkai adapters → core → ui; `core/` tidak meng-import `adapters/` atau `ui/`
- `models/` — `labels.json` (+ `exercise_classifier.onnx` setelah Tahap 1); tanpa model: pilih latihan manual
- `tests/unit/` mencerminkan `src/core/`; `tests/e2e/` smoke test Playwright; `tests/fixtures/` video & golden

```bash
npm start            # http://localhost:5173 (kamera hanya di localhost / HTTPS)
npm test && npm run test:e2e   # e2e juga memeriksa tiap request: video tidak pernah dikirim
```
