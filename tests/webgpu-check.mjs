// WebGPU-Backend gegen die CPU-Referenz prüfen (Headless-Chromium mit SwiftShader).
//   node tests/webgpu-check.mjs <playwright-core-pfad>
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.argv[2]);
const root = fileURLToPath(new URL('..', import.meta.url));
const types = { '.js': 'text/javascript', '.html': 'text/html', '.txt': 'text/plain' };
const server = http.createServer(async (req, res) => {
  if (req.url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>WebGPU-Check</title>'); return; }
  try {
    const body = await readFile(join(root, decodeURIComponent(req.url.split('?')[0])));
    res.writeHead(200, { 'content-type': types[extname(req.url)] || 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
}).listen(8796);

const browser = await chromium.launch({
  executablePath: `${process.env.HOME}/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome`,
  args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[Browser]', m.text()));
page.on('pageerror', (e) => console.log('[Fehler]', e.message));
await page.goto('http://localhost:8796/');

const result = await page.evaluate(async () => {
  const { createCpuBackend } = await import('/public/js/llama/backend-cpu.js');
  const { createWebGpuBackend } = await import('/public/js/llama/backend-webgpu.js');
  const { createLlama } = await import('/public/js/llama/model.js');
  const { makeRng } = await import('/public/js/rng.js');
  const gpu = await createWebGpuBackend();
  if (!gpu) return { error: 'kein WebGPU' };
  const configs = [
    { label: 'krumme Größen + GQA', cfg: { vocabSize: 50, dim: 32, nLayers: 2, nHeads: 4, nKvHeads: 2, headDim: 8, ffnDim: 48, ctx: 9, ropeBase: 10000, normEps: 1e-5 }, batch: 3 },
    { label: 'realistisch (head_dim 64)', cfg: { vocabSize: 300, dim: 128, nLayers: 2, nHeads: 2, nKvHeads: 1, headDim: 64, ffnDim: 384, ctx: 64, ropeBase: 10000, normEps: 1e-5 }, batch: 2 },
  ];
  const out = [];
  const { MATMUL_VARIANTS } = await import('/public/js/llama/backend-webgpu.js');
  for (const { label, cfg, batch } of configs) {
    for (let vi = 0; vi < MATMUL_VARIANTS.length; vi++) {
      gpu.setMatmulVariant({ NT: vi, NN: vi, TN: vi });
      const rng = makeRng(3);
      const N = batch * cfg.ctx;
      const inp = Int32Array.from({ length: N }, () => rng.int(cfg.vocabSize));
      const tgt = Int32Array.from({ length: N }, () => rng.int(cfg.vocabSize));
      const mc = createLlama(cfg, createCpuBackend(), { batch }), mg = createLlama(cfg, gpu, { batch });
      const p0 = mc.initialParams(5).map((x) => x * 5);
      for (const t of mc.layout.tensors) if (t.kind === 'norm') for (let i = 0; i < t.size; i++) p0[t.offset + i] = 0.5 + rng.random();
      mc.setParams(p0); mg.setParams(p0);
      mc.setBatch(inp, tgt); mg.setBatch(inp, tgt);
      const lc = await mc.lossAndGrads(), lg = await mg.lossAndGrads();
      const gc = await mc.readGrads(), gg = await mg.readGrads();
      const perTensor = mc.layout.tensors.map((t) => {
        let maxAbs = 0, maxDiff = 0;
        for (let i = t.offset; i < t.offset + t.size; i++) { maxAbs = Math.max(maxAbs, Math.abs(gc[i])); maxDiff = Math.max(maxDiff, Math.abs(gc[i] - gg[i])); }
        return { name: t.name, rel: maxDiff / (maxAbs || 1) };
      });
      const worst = perTensor.reduce((a, b) => (b.rel > a.rel ? b : a));
      // drei echte Lernschritte: CPU wartend, GPU mit Pipelining (queueStep)
      const gpuLosses = [];
      for (let s = 1; s <= 3; s++) {
        await mc.trainStep({ lr: 1e-3, wd: 0.1, t: s });
        gpuLosses.push(mg.queueStep({ lr: 1e-3, wd: 0.1, t: s }));
      }
      await Promise.all(gpuLosses);
      // Adam normiert jeden Schritt auf ~lr; bei Gradienten nahe 0 kann Float-Rauschen
      // die Richtung umdrehen. Aussagekräftig ist daher der Loss nach dem Training.
      const pc = await mc.readParams(), pg = await mg.readParams();
      let pDiff = 0;
      for (let i = 0; i < pc.length; i++) pDiff = Math.max(pDiff, Math.abs(pc[i] - pg[i]));
      const afterCpu = await mc.loss(), afterGpu = await mg.loss();
      out.push({ label: `${label}, Matmul-Variante ${vi}`, lossCpu: lc, lossGpu: lg, worst, pDiff, afterCpu, afterGpu });
    }
  }
  return { adapter: gpu.adapterInfo, out };
});
console.log(JSON.stringify(result, null, 1));
await browser.close();
server.close();
const ok = !result.error && result.out.every((r) => Math.abs(r.lossCpu - r.lossGpu) < 1e-4 && r.worst.rel < 2e-3 && r.pDiff < 6 * 1e-3 && Math.abs(r.afterCpu - r.afterGpu) < 1e-3);
console.log(ok ? 'GPU = CPU ✔' : 'ABWEICHUNG ✖');
process.exit(ok ? 0 : 1);
