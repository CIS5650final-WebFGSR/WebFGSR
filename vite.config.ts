import { defineConfig, type Plugin } from "vite";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));
const resultsRoot = path.join(projectRoot, "results");

/**
 * Dev-only endpoint used by the offline capture harness (tools/capture.mjs):
 *   POST /__save?path=<relative path under results/>   body = raw bytes
 * The page posts PNG frames / JSON measurements here so nothing large has to
 * cross the Playwright bridge.
 */
function saveResultsPlugin(): Plugin {
    return {
        name: "webfgsr-save-results",
        configureServer(server) {
            server.middlewares.use("/__save", (req, res) => {
                if (req.method !== "POST") {
                    res.statusCode = 405;
                    res.end("POST only");
                    return;
                }
                const url = new URL(req.url ?? "", "http://localhost");
                const rel = url.searchParams.get("path") ?? "";
                const target = path.resolve(resultsRoot, rel);
                if (!rel || !target.startsWith(resultsRoot + path.sep)) {
                    res.statusCode = 400;
                    res.end("bad path");
                    return;
                }
                const chunks: Buffer[] = [];
                req.on("data", (c: Buffer) => chunks.push(c));
                req.on("end", () => {
                    mkdir(path.dirname(target), { recursive: true })
                        .then(() => writeFile(target, Buffer.concat(chunks)))
                        .then(() => {
                            res.statusCode = 200;
                            res.end("ok");
                        })
                        .catch((e) => {
                            res.statusCode = 500;
                            res.end(String(e));
                        });
                });
            });
        },
    };
}

export default defineConfig({
    plugins: [saveResultsPlugin()],
    server: { port: 5173, strictPort: false, host: "127.0.0.1" },
    build: { target: "es2022", chunkSizeWarningLimit: 4000 },
});
