import { defineConfig } from '@playwright/test';

// Kept next to the spec rather than at the repo root (docs/PLAN.md §4: no loose root files).
export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.js',
  timeout: 120_000,
  reporter: 'list',
  use: {
    browserName: 'chromium',
    headless: true,
    launchOptions: { args: ['--autoplay-policy=no-user-gesture-required'] },
  },
});
