#!/usr/bin/env node
/**
 * Keeps the repository the shape docs/PLAN.md §4 maps out. Run by CI:
 *
 *   node tools/checkStructure.js
 *
 * Exits 1, listing every problem, when:
 *   - a tracked file or folder sits at the root but is not in the §4 map;
 *   - a file in app/src/core/ imports from app/src/adapters/ or app/src/ui/ (§4: core stays pure);
 *   - a source file is longer than 400 lines (§6.7);
 *   - a video, keypoint archive or model checkpoint is tracked outside the allowed places (§6.5).
 * Only files tracked by git are checked, so local data/, .venv/ and node_modules/ never count.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const ROOT_ENTRIES = Object.freeze([
  '.github', '.gitignore', 'README.md', 'package.json', 'package-lock.json', 'eslint.config.js',
  'app', 'ml', 'tools', 'labels', 'reports', 'docs',
]);
export const MAX_LINES = 400;

const SOURCE = /\.(js|mjs|py)$/;
const HEAVY = /\.(mp4|mov|avi|mkv|webm|npz|pt|pth|joblib|task|onnx)$/i;
// Small fixtures and the shipped classifier are the only heavy files allowed in git (§6.5).
const HEAVY_ALLOWED = [/^app\/tests\/fixtures\/videos\/[^/]+\.(mp4|webm)$/, /^app\/models\/exercise_classifier\.onnx$/];
const CORE = 'app/src/core/';
const FORBIDDEN_FROM_CORE = ['app/src/adapters/', 'app/src/ui/'];

// ---------------------------------------------------------------- pure checks

export function rootViolations(files) {
  const roots = [...new Set(files.map((file) => file.split('/')[0]))].sort();
  return roots.filter((entry) => !ROOT_ENTRIES.includes(entry))
    .map((entry) => `${entry}: not in the docs/PLAN.md §4 map of the root`);
}

/** Every module specifier a JS file imports, statically or dynamically. */
export function importSpecifiers(source) {
  const patterns = [/\bimport\s[^'"]*?from\s*['"]([^'"]+)['"]/g, /\bexport\s[^'"]*?from\s*['"]([^'"]+)['"]/g,
    /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g, /\bimport\s*['"]([^'"]+)['"]/g];
  return [...new Set(patterns.flatMap((pattern) => [...source.matchAll(pattern)].map((match) => match[1])))];
}

/** Imports of a core file that resolve into adapters/ or ui/ (relative paths are resolved, not pattern-matched). */
export function forbiddenImports(file, source) {
  if (!file.startsWith(CORE) || !/\.m?js$/.test(file)) return [];
  return importSpecifiers(source)
    .filter((specifier) => specifier.startsWith('.'))
    .map((specifier) => path.posix.normalize(path.posix.join(path.posix.dirname(file), specifier)))
    .filter((target) => FORBIDDEN_FROM_CORE.some((prefix) => target.startsWith(prefix)))
    .map((target) => `${file}: imports ${target} — app/src/core/ must not depend on adapters/ or ui/`);
}

export function longFiles(entries, maxLines = MAX_LINES) {
  return entries.filter(({ file, lines }) => SOURCE.test(file) && lines > maxLines)
    .map(({ file, lines }) => `${file}: ${lines} lines, over the ${maxLines}-line limit (docs/PLAN.md §6.7)`);
}

export function heavyFiles(files) {
  return files.filter((file) => HEAVY.test(file) && !HEAVY_ALLOWED.some((allowed) => allowed.test(file)))
    .map((file) => `${file}: videos, keypoints and checkpoints stay out of git (docs/PLAN.md §6.5)`);
}

// ---------------------------------------------------------------- I/O

function trackedFiles(cwd) {
  return execFileSync('git', ['ls-files', '-z'], { cwd, encoding: 'utf8' }).split('\0').filter(Boolean);
}

export function checkRepository(root) {
  const files = trackedFiles(root);
  const read = (file) => readFileSync(path.join(root, file), 'utf8');
  const sources = files.filter((file) => SOURCE.test(file));
  return [
    ...rootViolations(files),
    ...sources.flatMap((file) => forbiddenImports(file, read(file))),
    ...longFiles(sources.map((file) => ({ file, lines: read(file).split('\n').length - 1 }))),
    ...heavyFiles(files),
  ];
}

function main() {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const problems = checkRepository(root);
  if (problems.length) {
    console.error(`${problems.length} structure problem(s):\n${problems.map((p) => `  - ${p}`).join('\n')}`);
    return 1;
  }
  console.log('structure ok: root entries, core imports, file lengths and tracked binaries all match docs/PLAN.md');
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = main();
