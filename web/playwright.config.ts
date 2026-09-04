import { defineConfig, devices } from "@playwright/test";

const PORT = 4173;
const SERVER_PORT = 8080;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
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
      use: { ...devices["Desktop Chrome"] },
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
