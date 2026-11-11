#!/usr/bin/env node
// WebFGSR offline driver: starts the Vite dev server, launches desktop Chrome
// (headless, WebGPU on) through playwright-core and drives the page.
//
//   node tools/capture.mjs smoke   [--views split,sr,motion] [--wait 4] [--size 1920x1080] [--query k=v&...]
//   node tools/capture.mjs capture --run sponza_walk [--series sr,bilinear,native,reference]
//                                  [--frames 240] [--fps 60] [--t0 0] [--warmup 32] [--save-every 1]
//                                  [--size 1920x1080] [--settings '{"jitterSign":-1}'] [--prefix name_]
//   node tools/capture.mjs bench   [--size 1920x1080] [--frames 300] [--warmup 60]
//
// Frames land in results/<run>/frames/<series>/NNNN.png (written by the dev
// server's /__save endpoint, see vite.config.ts).

import { chromium } from "playwright-core";
import { createServer } from "vite";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultsDir = path.join(root, "results");

function parseArgs(argv) {
    const [command, ...rest] = argv;
    const opts = {};
    for (let i = 0; i < rest.length; i++) {
        const a = rest[i];
        if (!a.startsWith("--")) {
            throw new Error(`unexpected argument ${a}`);
        }
        const key = a.slice(2);
        const next = rest[i + 1];
        if (next === undefined || next.startsWith("--")) {
            opts[key] = true;
        } else {
            opts[key] = next;
            i++;
        }
    }
    return { command, opts };
}

const HELP = `usage:
  node tools/capture.mjs smoke   [--views split,sr,motion,depth,weight,lr,native,bilinear] [--wait 4] [--size WxH] [--query "k=v&k2=v2"]
  node tools/capture.mjs capture --run NAME [--series sr,bilinear,native,reference] [--frames 240] [--fps 60]
                                 [--t0 0] [--warmup 32] [--save-every 1] [--size WxH] [--settings JSON] [--prefix STR] [--query STR]
  node tools/capture.mjs bench   [--size WxH] [--frames 300] [--warmup 60] [--query STR]
  --headed  show the browser window`;

async function startServer() {
    const server = await createServer({ root, configFile: path.join(root, "vite.config.ts"), logLevel: "warn", server: { port: 0, host: "127.0.0.1" } });
    await server.listen();
    const url = server.resolvedUrls?.local?.[0];
    if (!url) {
        throw new Error("vite did not report a URL");
    }
    return { server, url };
}

async function launch(headed) {
    return chromium.launch({
        channel: "chrome",
        headless: !headed,
        args: [
            "--enable-unsafe-webgpu",
            "--enable-webgpu-developer-features", // full-precision timestamp queries
            "--ignore-gpu-blocklist",
            "--disable-gpu-vsync",
            "--disable-frame-rate-limit",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
        ],
    });
}

async function openPage(browser, url, query, size, log) {
    const [w, h] = size;
    const page = await browser.newPage({ viewport: { width: Math.min(w, 1920), height: Math.min(h, 1080) }, deviceScaleFactor: 1 });
    const external = new Set();
    const errors = [];
    page.on("request", (req) => {
        const u = new URL(req.url());
        if (u.hostname !== "127.0.0.1" && u.hostname !== "localhost" && u.protocol.startsWith("http")) {
            external.add(req.url());
        }
    });
    page.on("console", (msg) => {
        const text = msg.text();
        if (msg.type() === "error" || msg.type() === "warning") {
            errors.push(`${msg.type()}: ${text}`);
        }
        if (log) {
            console.log(`  [page:${msg.type()}] ${text}`);
        }
    });
    page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
    await page.goto(`${url}?${query}`, { waitUntil: "load" });
    await page.waitForFunction(() => window.webfgsr?.ready === true, null, { timeout: 120_000 });
    return { page, external, errors };
}

function reportPageIssues(name, external, errors) {
    if (external.size) {
        console.log(`  ${name}: external requests: ${[...external].join(", ")}`);
    }
    const filtered = errors.filter((e) => !/DevTools|Download the React|lil-gui/.test(e));
    if (filtered.length) {
        console.log(`  ${name}: ${filtered.length} console errors/warnings:`);
        for (const e of filtered.slice(0, 20)) {
            console.log(`    ${e.slice(0, 400)}`);
        }
    }
}

async function smoke(browser, url, opts) {
    const views = String(opts.views ?? "split,sr,bilinear,native,lr,motion,depth,weight").split(",");
    const wait = Number(opts.wait ?? 4);
    const size = parseSize(opts.size);
    const outDir = path.join(resultsDir, "smoke");
    await mkdir(outDir, { recursive: true });
    for (const view of views) {
        const q = `view=${view}&out=${size[0]}x${size[1]}${opts.query ? `&${opts.query}` : ""}`;
        const { page, external, errors } = await openPage(browser, url, q, size, false);
        await page.waitForTimeout(wait * 1000);
        const hud = await page.evaluate(() => document.getElementById("hud")?.textContent ?? "");
        const file = path.join(outDir, `${view}.png`);
        await page.locator("#renderCanvas").screenshot({ path: file });
        console.log(`smoke ${view}: ${path.relative(root, file)}`);
        console.log(hud.split("\n").map((l) => `    ${l}`).join("\n"));
        reportPageIssues(view, external, errors);
        await page.close();
    }
}

function parseSize(v) {
    const m = String(v ?? "1920x1080").match(/^(\d+)x(\d+)$/);
    if (!m) {
        throw new Error(`bad --size ${v}`);
    }
    return [Number(m[1]), Number(m[2])];
}

async function capture(browser, url, opts) {
    if (!opts.run) {
        throw new Error("--run is required");
    }
    const size = parseSize(opts.size);
    const q = `capture=1&out=${size[0]}x${size[1]}${opts.query ? `&${opts.query}` : ""}`;
    const { page, external, errors } = await openPage(browser, url, q, size, true);
    const info = await page.evaluate(() => window.webfgsr.adapterInfo());
    console.log(`adapter: ${JSON.stringify(info)}`);
    const settings = opts.settings ? JSON.parse(opts.settings) : null;
    const warmFrames = await page.evaluate(async (s) => {
        if (s) {
            window.webfgsr.setSettings(s);
        }
        return window.webfgsr.warmUp();
    }, settings);
    console.log(`ready after ${warmFrames} warm-up frames`);
    const captureOpts = {
        run: String(opts.run),
        series: String(opts.series ?? "sr,bilinear,native,reference").split(","),
        frames: Number(opts.frames ?? 240),
        fps: Number(opts.fps ?? 60),
        t0: Number(opts.t0 ?? 0),
        warmupFrames: Number(opts.warmup ?? 32),
        saveEvery: Number(opts["save-every"] ?? 1),
        prefix: opts.prefix ? String(opts.prefix) : undefined,
    };
    const t = Date.now();
    const meta = await page.evaluate((o) => window.webfgsr.capture(o), captureOpts);
    meta.adapter = info;
    meta.settingsOverride = settings;
    await writeFile(path.join(resultsDir, captureOpts.run, `capture_${captureOpts.prefix ?? ""}meta.json`), JSON.stringify(meta, null, 2));
    console.log(`captured ${meta.savedFrames} frames x ${captureOpts.series.length} series in ${((Date.now() - t) / 1000).toFixed(1)} s`);
    reportPageIssues("capture", external, errors);
    await page.close();
}

async function bench(browser, url, opts) {
    const size = parseSize(opts.size);
    const q = `capture=1&out=${size[0]}x${size[1]}${opts.query ? `&${opts.query}` : ""}`;
    const { page, external, errors } = await openPage(browser, url, q, size, false);
    const info = await page.evaluate(() => window.webfgsr.adapterInfo());
    await page.evaluate(() => window.webfgsr.warmUp());
    const frames = Number(opts.frames ?? 300);
    const warmupFrames = Number(opts.warmup ?? 60);
    const configs = [
        { label: "sr", series: ["sr"] },
        { label: "native", series: ["native"] },
        { label: "bilinear", series: ["bilinear"] },
    ];
    const results = [];
    for (const c of configs) {
        const r = await page.evaluate((o) => window.webfgsr.benchmark(o), { ...c, frames, warmupFrames, fps: 60, t0: 0 });
        results.push(r);
        const passes = Object.entries(r.passes)
            .map(([k, v]) => `${k}=${v.ms.toFixed(3)}ms`)
            .join("  ");
        console.log(`bench ${size[0]}x${size[1]} ${c.label}: ${passes}`);
    }
    const out = { adapter: info, size, frames, results };
    const file = path.join(resultsDir, "bench", `bench_${size[0]}x${size[1]}${opts.tag ? `_${opts.tag}` : ""}.json`);
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, JSON.stringify(out, null, 2));
    console.log(`wrote ${path.relative(root, file)}`);
    reportPageIssues("bench", external, errors);
    await page.close();
}

async function main() {
    const { command, opts } = parseArgs(process.argv.slice(2));
    if (!command || command === "--help" || opts.help) {
        console.log(HELP);
        process.exit(command ? 0 : 1);
    }
    const { server, url } = await startServer();
    const browser = await launch(!!opts.headed);
    try {
        if (command === "smoke") {
            await smoke(browser, url, opts);
        } else if (command === "capture") {
            await capture(browser, url, opts);
        } else if (command === "bench") {
            await bench(browser, url, opts);
        } else {
            throw new Error(`unknown command ${command}\n${HELP}`);
        }
    } finally {
        await browser.close();
        await server.close();
    }
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
