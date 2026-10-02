import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Smoke test of the real app in Chromium: MediaPipe runs on the fixture push-up video and the count
 * goes above zero, with no console errors and no request that could carry a frame off the device.
 *
 *   npm run test:e2e        (needs `python -m repcount.data.download_model` once, for the pose model)
 *
 * The page is served straight from app/ through request routing, and the MediaPipe bundle is served
 * from the npm package @mediapipe/tasks-vision — the same files the app loads from cdn.jsdelivr.net in
 * production — so the test runs without a web server and without the CDN.
 */

const ROOT = path.resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const APP = path.join(ROOT, 'app');
const MEDIAPIPE_DIR = path.join(ROOT, 'node_modules', '@mediapipe', 'tasks-vision');
const POSE_MODEL = path.join(ROOT, 'data', 'pose_models', 'pose_landmarker_lite.task');
const VIDEOS = path.join(APP, 'tests', 'fixtures', 'videos');
const ORIGIN = 'http://localhost:4173'; // localhost: a secure context, as with `npm start`
const MEDIAPIPE_PREFIX = '/npm/@mediapipe/tasks-vision@1.0.1/';
// Every host the app may contact: itself, the MediaPipe library CDN and the pose model's bucket.
const ALLOWED_HOSTS = new Set(['localhost:4173', 'cdn.jsdelivr.net', 'storage.googleapis.com']);

const TYPES = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.task': 'application/octet-stream',
};

async function fulfillFile(route, file) {
  if (!existsSync(file)) return route.fulfill({ status: 404, body: '' });
  return route.fulfill({ status: 200, body: await readFile(file),
    headers: { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream' } });
}

/** Serve the app and record every request the page makes, for the privacy check. */
async function serve(page) {
  const requests = [];
  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push({ method: request.method(), host: url.host, path: url.pathname, bodyBytes: request.postDataBuffer()?.length ?? 0 });
    if (url.origin === ORIGIN) return fulfillFile(route, path.join(APP, url.pathname));
    if (url.host === 'cdn.jsdelivr.net' && url.pathname.startsWith(MEDIAPIPE_PREFIX)) {
      return fulfillFile(route, path.join(MEDIAPIPE_DIR, url.pathname.slice(MEDIAPIPE_PREFIX.length)));
    }
    if (url.host === 'storage.googleapis.com' && url.pathname.endsWith('pose_landmarker_lite.task') && existsSync(POSE_MODEL)) {
      return fulfillFile(route, POSE_MODEL);
    }
    return ALLOWED_HOSTS.has(url.host) ? route.continue() : route.abort();
  });
  return requests;
}

// MediaPipe's WASM routes its whole log to console.error, including INFO lines such as
// "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." Only those are set aside; every other
// console error, and every uncaught exception, fails the test.
const MEDIAPIPE_INFO = /^INFO: /;

function watchConsole(page) {
  const errors = [];
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error' && !MEDIAPIPE_INFO.test(message.text())) errors.push(`console: ${message.text()}`);
  });
  return errors;
}

/** Real Chrome plays the H.264 original; Playwright's open-source Chromium has no H.264, so it gets the WebM copy. */
async function fixtureVideo(page) {
  const h264 = await page.evaluate(() => document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"'));
  return path.join(VIDEOS, h264 ? 'push-up_17.mp4' : 'push-up_17.webm');
}

async function openApp(page) {
  const requests = await serve(page);
  const errors = watchConsole(page);
  await page.goto(`${ORIGIN}/index.html`);
  await expect(page.locator('#status')).toHaveText(/Model siap/, { timeout: 60_000 });
  return { requests, errors };
}

test('the push-up video is counted, with no console errors and no frame leaving the device', async ({ page }) => {
  const { requests, errors } = await openApp(page);
  await page.setInputFiles('#video-file', await fixtureVideo(page));

  await expect.poll(async () => Number(await page.textContent('#count')), { timeout: 90_000, intervals: [500] })
    .toBeGreaterThan(0);

  // No trained classifier yet: the app says so, offers the manual choice, and still counts.
  await expect(page.locator('#model-status')).toBeVisible();
  await expect(page.locator('#model-status')).toContainText('belum tersedia');
  expect(errors).toEqual([]);

  // Privacy: only downloads (GET, no body) from the expected hosts — nothing that could carry a frame.
  expect(requests.filter((r) => r.method !== 'GET')).toEqual([]);
  expect(requests.filter((r) => r.bodyBytes > 0)).toEqual([]);
  expect([...new Set(requests.map((r) => r.host))].filter((host) => !ALLOWED_HOSTS.has(host))).toEqual([]);
  expect(requests.filter((r) => r.path.includes('onnxruntime')), 'no runtime download without a model').toEqual([]);
});

test('choosing an exercise by hand switches the panel to it', async ({ page }) => {
  const { errors } = await openApp(page);
  await page.setInputFiles('#video-file', await fixtureVideo(page));

  await page.selectOption('#exercise', 'push_up');
  await expect(page.locator('#phase')).toHaveText('Menghitung');
  await expect(page.locator('#exercise-name')).toHaveText('Push-up');
  await expect(page.locator('#exercise-meta')).toContainText('eksperimental');

  await page.selectOption('#exercise', 'plank');
  await expect(page.locator('#hold-card')).toBeVisible();
  await expect(page.locator('#reps-card')).toBeHidden();

  await page.selectOption('#exercise', 'lat_pulldown');
  await expect(page.locator('#form-note')).toContainText('Belum ada aturan form');
  expect(errors).toEqual([]);
});
