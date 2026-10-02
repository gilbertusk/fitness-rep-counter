import js from '@eslint/js';
import globals from 'globals';

// Minimal on purpose: ESLint's recommended rules, with browser or Node globals where each part runs.
export default [
  { ignores: ['node_modules/', 'data/', '.venv/', 'ml/', 'test-results/', 'playwright-report/'] },
  js.configs.recommended,
  {
    files: ['**/*.js'],
    languageOptions: { ecmaVersion: 2024, sourceType: 'module', globals: { ...globals.browser } },
  },
  {
    // Command-line tools, tests and configs run in Node.
    files: ['tools/eval/**/*.js', 'tools/checkStructure*.js', '**/tests/**/*.js', '**/*.test.js', '*.config.js'],
    languageOptions: { globals: { ...globals.node } },
  },
];
