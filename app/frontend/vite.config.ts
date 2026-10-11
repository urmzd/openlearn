import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

/**
 * main.go embeds frontend/dist, and //go:embed fails outright on a missing
 * directory, so the repo tracks dist/.gitkeep to keep a fresh clone buildable
 * before anyone runs a frontend build.
 *
 * vite empties the output directory on every build and takes .gitkeep with it,
 * which silently stages a deletion. Writing it back on close means neither
 * `wails dev` nor `wails build` nor a bare `npm run build` can lose it.
 */
function keepDistTracked(): Plugin {
  return {
    name: "whiteboardy:keep-dist-tracked",
    apply: "build",
    closeBundle() {
      writeFileSync(resolve(__dirname, "dist/.gitkeep"), "");
    },
  };
}

export default defineConfig({
  plugins: [react(), keepDistTracked()],
});
