# app/ — Web app (yang di-deploy)

Rep counter di browser: MediaPipe Pose Lite → sudut sendi → penghitung repetisi + aturan form.

- `index.html`, `styles.css` — halaman
- `src/core/` — logika murni tanpa DOM (geometry, counting, form, exercises), 100% di-test
- `src/adapters/` — pembungkus browser API (MediaPipe Pose Landmarker)
- `src/ui/` — render DOM & kanvas (overlay kerangka, statistik)
- `tests/unit/` — unit test, mencerminkan `src/core/`; `tests/fixtures/videos/` — video uji

```bash
npm start   # http://localhost:5173
npm test
```
