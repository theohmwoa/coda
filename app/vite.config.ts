import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Output goes into coda's package data so `python -m coda serve` can
// serve it from the same wheel/checkout.
const codaStaticApp = path.resolve(
  __dirname, "..", "src", "coda", "static", "app"
);

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: codaStaticApp,
    emptyOutDir: true,
    sourcemap: false,
    // The Python server serves this under /app/* — use relative asset paths
    // so the built index.html loads correctly regardless of mount point.
    assetsDir: "assets",
  },
  // Served from `/` by the FastAPI server, with `/assets/*` mounted
  // directly onto static/app/assets/. Absolute base means `index.html`
  // requests `/assets/...` which the server resolves correctly.
  base: "/",
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true, changeOrigin: true },
      "/health": "http://127.0.0.1:8000",
      "/skills": "http://127.0.0.1:8000",
    },
  },
});
