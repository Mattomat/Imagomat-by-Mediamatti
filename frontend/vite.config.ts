import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Entwicklung: Vite auf 1420, API-Aufrufe gehen an das Python-Backend auf 8765.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", ws: true },
    },
  },
  build: { outDir: "dist", target: "es2021" },
});
