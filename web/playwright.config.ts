import { defineConfig, devices } from "@playwright/test";

// Ports are overridable so the suite can be run on a machine that already has
// something on the defaults (a `docker compose up` of this same app, say)
// without editing this file. `VITE_API_BASE` is baked into the bundle at
// build time, so overriding `DUELS_E2E_SERVER_PORT` means rebuilding with a
// matching `VITE_API_BASE`.
const PORT = Number(process.env.DUELS_E2E_PORT ?? 4173);
const SERVER_PORT = Number(process.env.DUELS_E2E_SERVER_PORT ?? 8080);

export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: [
    {
      // The real duels-server, built in release mode by the CI job before
      // this runs. `DUELS_SERVER_ADDR` matches `crates/duels-server`'s
      // `main.rs`.
      command: `DUELS_SERVER_ADDR=127.0.0.1:${SERVER_PORT} ../target/release/duels-server`,
      url: `http://127.0.0.1:${SERVER_PORT}/catalog`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      // Serves the already-built `dist/` (built with `npm run build` before
      // this runs, in CI). `VITE_API_BASE` is a build-time constant baked
      // into the bundle, so it can't be overridden here - it defaults to
      // `http://localhost:8080`, which matches `duels-server`'s own default
      // address, so no override is needed.
      command: `npm run preview -- --port ${PORT}`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
