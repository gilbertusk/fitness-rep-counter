import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ROOT_ENTRIES, rootViolations, importSpecifiers, forbiddenImports, longFiles, heavyFiles, checkRepository,
} from './checkStructure.js';

test('only the root entries of the §4 map are allowed', () => {
  assert.deepEqual(rootViolations(['README.md', 'app/index.html', 'ml/README.md', '.github/workflows/ci.yml']), []);
  assert.deepEqual(rootViolations(['notes.txt', 'app/x.js', 'scratch/a.py']), [
    'notes.txt: not in the docs/PLAN.md §4 map of the root',
    'scratch: not in the docs/PLAN.md §4 map of the root',
  ]);
  assert.ok(ROOT_ENTRIES.includes('package-lock.json'));
});

test('static, re-exported, dynamic and side-effect imports are all found', () => {
  const source = `import a from './a.js';
import { b,
  c } from "../b.js";
export { d } from './d.js';
const e = await import('./e.js');
import './f.js';`;
  assert.deepEqual(importSpecifiers(source).sort(), ['../b.js', './a.js', './d.js', './e.js', './f.js']);
});

test('core may import core, but never adapters/ or ui/', () => {
  assert.deepEqual(forbiddenImports('app/src/core/session/workout.js', "import { x } from '../form/measure.js';"), []);
  const bad = forbiddenImports('app/src/core/session/workout.js', "import { speak } from '../../adapters/speech.js';");
  assert.equal(bad.length, 1);
  assert.match(bad[0], /imports app\/src\/adapters\/speech\.js/);
  assert.equal(forbiddenImports('app/src/core/x.js', "const ui = await import('../ui/panel.js');").length, 1);
});

test('the import rule resolves paths instead of matching text', () => {
  // '../core/../ui/x.js' contains no "../ui/" prefix at first glance but still lands in ui/.
  assert.equal(forbiddenImports('app/src/core/a.js', "import x from '../core/../ui/x.js';").length, 1);
  // A file *named* like ui inside core is fine.
  assert.deepEqual(forbiddenImports('app/src/core/a.js', "import x from './ui-helpers.js';"), []);
  // Outside core the rule does not apply: main.js wires everything together.
  assert.deepEqual(forbiddenImports('app/src/main.js', "import x from './adapters/speech.js';"), []);
});

test('source files over 400 lines are reported, other files are not', () => {
  assert.deepEqual(longFiles([{ file: 'a.js', lines: 400 }, { file: 'b.json', lines: 9000 }]), []);
  assert.match(longFiles([{ file: 'ml/x.py', lines: 401 }])[0], /401 lines/);
});

test('videos and checkpoints stay out of git, except the small fixtures and the shipped model', () => {
  assert.deepEqual(heavyFiles([
    'app/tests/fixtures/videos/push-up_17.mp4', 'app/tests/fixtures/videos/push-up_17.webm',
    'app/models/exercise_classifier.onnx', 'ml/splits/split_v1.json',
  ]), []);
  assert.equal(heavyFiles(['data/keypoints/squat/squat_1.npz', 'clip.MOV', 'run/temporal.pt']).length, 3);
});

test('this repository passes its own structure check', () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  assert.deepEqual(checkRepository(root), []);
});
