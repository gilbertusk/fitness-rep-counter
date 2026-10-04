/**
 * Pure helpers of the benchmark: summary statistics, asset grouping and the Markdown report.
 * No DOM here, so all of it is tested in Node (tools/benchmark/tests/stats.test.js).
 */

/** Percentile with linear interpolation between closest ranks (p in [0, 100]); NaN for no samples. */
export function percentile(values, p) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const rank = (p / 100) * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
}

/** @returns {{ n: number, median: number, p95: number, mean: number, max: number }} times in ms */
export function summarize(values) {
  const finite = values.filter(Number.isFinite);
  return {
    n: finite.length,
    median: percentile(finite, 50),
    p95: percentile(finite, 95),
    mean: finite.length ? finite.reduce((sum, v) => sum + v, 0) / finite.length : NaN,
    max: finite.length ? Math.max(...finite) : NaN,
  };
}

/** Frames per second over a run: (frames - 1) intervals between the first and the last frame. */
export function framesPerSecond(frameTimesMs) {
  if (frameTimesMs.length < 2) return NaN;
  const span = frameTimesMs[frameTimesMs.length - 1] - frameTimesMs[0];
  return span > 0 ? ((frameTimesMs.length - 1) * 1000) / span : NaN;
}

const ASSET_GROUPS = [
  ['Model pose (.task)', /\.task$/],
  ['MediaPipe WASM', /tasks-vision.*\.wasm$/],
  ['MediaPipe JS', /tasks-vision.*\.(m?js)$/],
  ['ONNX Runtime WASM', /onnxruntime-web.*\.wasm$/],
  ['ONNX Runtime JS', /onnxruntime-web.*\.(m?js)$/],
  ['Model pengenal (.onnx)', /\.onnx$/],
  ['Kode app (JS/CSS/HTML/JSON)', /\/app\/.*\.(m?js|css|html|json)$/],
];

/**
 * Sizes per group from PerformanceResourceTiming-like entries. A size of 0 with a non-zero duration
 * means the browser was not allowed to read it (no Timing-Allow-Origin) or it came from cache;
 * such entries are counted in `unknown` rather than as 0 bytes.
 * @param {{ name: string, encodedBodySize: number, decodedBodySize: number }[]} entries
 */
export function groupAssets(entries) {
  const groups = ASSET_GROUPS.map(([label]) => ({ label, files: 0, encodedBytes: 0, decodedBytes: 0, unknown: 0 }));
  for (const entry of entries) {
    const url = entry.name.split('?')[0];
    const index = ASSET_GROUPS.findIndex(([, pattern]) => pattern.test(url));
    if (index < 0) continue;
    const group = groups[index];
    group.files += 1;
    if (entry.decodedBodySize > 0) {
      group.encodedBytes += entry.encodedBodySize;
      group.decodedBytes += entry.decodedBodySize;
    } else {
      group.unknown += 1;
    }
  }
  return groups.filter((group) => group.files > 0);
}

const ms = (value) => (Number.isFinite(value) ? value.toFixed(value < 1 ? 3 : 1) : '–');
const mb = (bytes) => (bytes / 1_000_000).toFixed(2);

/**
 * The report pasted into reports/05-performance/performance.md.
 * @param {object} result what bench.js collects (see its runBenchmark)
 */
export function toMarkdown(result) {
  const { device, load, live, stages, assets, video } = result;
  const lines = [
    `### ${device.label || 'Perangkat'} — ${new Date(result.dateMs).toISOString().slice(0, 10)}`,
    '',
    `- Browser: \`${device.userAgent}\``,
    `- CPU logis: ${device.cores ?? '–'} · Delegate MediaPipe: **${device.delegate}** · WebGL: \`${device.renderer ?? 'tidak ada'}\``,
    `- Video: ${video.name} (${video.width}×${video.height}, ${video.durationS.toFixed(1)} s × ${video.repeat} putaran)`,
    '',
    '| Ukuran | Nilai |',
    '|---|---|',
    `| FPS end-to-end (pose + core + overlay) | **${Number.isFinite(live.fps) ? live.fps.toFixed(1) : '–'}** (dibatasi FPS video ≈ ${video.fps ?? '–'}) |`,
    `| Frame diproses | ${live.frames} (${live.posesFound} dengan pose) |`,
    `| Muat model pose (dari buka halaman) | ${ms(load.poseReadyMs)} ms |`,
    `| Pose pertama terdeteksi (dari buka halaman) | ${ms(load.firstPoseMs)} ms |`,
    `| Muat pengenal latihan | ${load.classifierReadyMs == null ? 'tidak dimuat' : `${ms(load.classifierReadyMs)} ms`} |`,
    '',
    '| Tahap (ms per panggilan) | n | median | p95 | maks |',
    '|---|---:|---:|---:|---:|',
    ...stages.map(({ label, summary }) =>
      `| ${label} | ${summary.n} | ${ms(summary.median)} | ${ms(summary.p95)} | ${ms(summary.max)} |`),
  ];
  if (assets.length) {
    lines.push('', '| Aset | File | Unduh (MB) | Setelah dekompresi (MB) |', '|---|---:|---:|---:|',
      ...assets.map((a) => `| ${a.label} | ${a.files} | ${a.unknown ? `${mb(a.encodedBytes)} + ${a.unknown} tak terbaca` : mb(a.encodedBytes)} | ${mb(a.decodedBytes)} |`));
  }
  if (result.notes?.length) lines.push('', ...result.notes.map((note) => `> ${note}`));
  return `${lines.join('\n')}\n`;
}
