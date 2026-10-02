#!/usr/bin/env node
/**
 * Runs tools/benchmark/ in Playwright's headless Chromium and prints the Markdown report, plus the
 * cold-cache first-load time of the real app (app/index.html until "Model siap").
 *
 *   node tools/benchmark/headless.js [--repeat 3] [--label "CI runner"] [--model x.onnx --labels labels.json]
 *
 * Pages are served from the repo through request routing (no web server). The CDN files come from
 * node_modules when installed there (@mediapipe/tasks-vision; onnxruntime-web is optional:
 * `npm install --no-save onnxruntime-web@1.23.2`) and the pose model from data/pose_models/ when
 * downloaded; anything missing is fetched from the network as in production.
 * Headless Chromium has no GPU: this measures the CPU path. Phones and laptops: see README.md.
 */
import { chromium } from '@playwright/test';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
const ORIGIN = 'http://localhost:5174';
const SERVED = ['app/', 'tools/benchmark/'];
const LOCAL_CDN = [
  ['/npm/@mediapipe/tasks-vision@1.0.1/', path.join(ROOT, 'node_modules/@mediapipe/tasks-vision/')],
  ['/npm/onnxruntime-web@1.23.2/', path.join(ROOT, 'node_modules/onnxruntime-web/')],
];
const POSE_MODEL = path.join(ROOT, 'data/pose_models/pose_landmarker_lite.task');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.webm': 'video/webm', '.mp4': 'video/mp4' };

const { values: args } = parseArgs({ options: {
  repeat: { type: 'string', default: '3' }, label: { type: 'string', default: 'Headless Chromium (CPU)' },
  model: { type: 'string' }, labels: { type: 'string' },
} });

async function fulfill(route, file) {
  if (!existsSync(file)) return route.fulfill({ status: 404, body: '' });
  return route.fulfill({ status: 200, body: await readFile(file),
    headers: { 'content-type': TYPES[path.extname(file)] ?? 'application/octet-stream', 'timing-allow-origin': '*' } });
}

async function route(context) {
  await context.route('**/*', async (r) => {
    const url = new URL(r.request().url());
    const relative = url.pathname.slice(1);
    if (url.origin === ORIGIN) {
      if (relative === 'bench-model/model.onnx' && args.model) return fulfill(r, path.resolve(args.model));
      if (relative === 'bench-model/labels.json' && args.labels) return fulfill(r, path.resolve(args.labels));
      const file = relative.endsWith('/') ? `${relative}index.html` : relative;
      return SERVED.some((prefix) => file.startsWith(prefix)) ? fulfill(r, path.join(ROOT, file)) : r.fulfill({ status: 404, body: '' });
    }
    if (url.host === 'cdn.jsdelivr.net') {
      const local = LOCAL_CDN.find(([prefix, dir]) => url.pathname.startsWith(prefix) && existsSync(dir));
      if (local) return fulfill(r, path.join(local[1], url.pathname.slice(local[0].length)));
    }
    if (url.pathname.endsWith('pose_landmarker_lite.task') && existsSync(POSE_MODEL)) return fulfill(r, POSE_MODEL);
    return r.continue();
  });
}

async function withPage(browser, use) {
  const context = await browser.newContext(); // fresh context = cold cache
  await route(context);
  const page = await context.newPage();
  try {
    return await use(page);
  } finally {
    await context.close();
  }
}

async function appFirstLoad(browser) {
  return withPage(browser, async (page) => {
    const start = Date.now();
    await page.goto(`${ORIGIN}/app/`);
    await page.waitForFunction(() => /Model siap/.test(document.getElementById('status')?.textContent ?? ''), null, { timeout: 120_000 });
    return Date.now() - start;
  });
}

async function benchmark(browser) {
  const query = new URLSearchParams({ autostart: '1', repeat: args.repeat, label: args.label });
  if (args.model) query.set('model', '/bench-model/model.onnx');
  if (args.labels) query.set('labels', '/bench-model/labels.json');
  return withPage(browser, async (page) => {
    await page.goto(`${ORIGIN}/tools/benchmark/?${query}`);
    await page.waitForFunction(() => window.__benchResult, null, { timeout: 600_000 });
    return page.evaluate(() => window.__benchResult);
  });
}

const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
try {
  const firstLoadMs = await appFirstLoad(browser);
  const bench = await benchmark(browser);
  if (bench.error) throw new Error(bench.error);
  process.stdout.write(bench.markdown);
  process.stdout.write(`\n> App: buka halaman → "Model siap" (cache dingin): ${firstLoadMs} ms.\n`);
} finally {
  await browser.close();
}
