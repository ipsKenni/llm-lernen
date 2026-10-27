// =====================================================================
// backend-webgpu.js – Dasselbe Rechenwerk auf der Grafikkarte
// =====================================================================
//
// Jede Operation aus backend-cpu.js gibt es hier als "Compute Shader" in
// WGSL, der Shader-Sprache von WebGPU. Die GPU startet tausende kleine
// Programme gleichzeitig: eins pro Ausgabewert, pro Zeile oder pro
// Kachel. Alle Daten bleiben auf der GPU; nur der Loss wird pro Schritt
// zurückgelesen.
//
// Geprüft wird jeder Shader gegen die CPU-Version (tests/webgpu-check.html).

const WG = 256;  // Threads pro Workgroup für einfache Operationen

// Ein Element-Index, der auch bei mehr als 65535 Workgroups funktioniert.
const IDX = `let i = gid.x + gid.y * nwg.x * ${WG}u;`;
const ELEM_ARGS = '@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u';

//#region webgpu-matmul
// Gekachelte Matrixmultiplikation: Jede Workgroup berechnet eine 64×64-Kachel
// von C, jeder ihrer 256 Threads davon 4×4 Werte. A- und B-Stücke werden
// gemeinsam in den schnellen Workgroup-Speicher geladen und von allen
// Threads wiederverwendet – das ist der wichtigste Trick schneller GPUs.
function matmulShader(transA, transB) {
  const aAt = transA ? 'A[gk * p.M + gm]' : 'A[gm * p.K + gk]';
  const bAt = transB ? 'B[gn * p.K + gk]' : 'B[gk * p.N + gn]';
  // Lade-Reihenfolge so, dass benachbarte Threads benachbarte Adressen lesen.
  const aIdx = transA ? 'let m = e % 64u; let k = e / 64u;' : 'let m = e / 16u; let k = e % 16u;';
  const bIdx = transB ? 'let n = e / 16u; let k = e % 16u;' : 'let n = e % 64u; let k = e / 64u;';
  return /* wgsl */ `
struct P { M: u32, N: u32, K: u32, beta: f32 }
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> As: array<f32, 1024>;  // [16][64]
var<workgroup> Bs: array<f32, 1024>;  // [16][64]
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  let tid = lid.y * 16u + lid.x;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var r = 0u; r < 4u; r++) {
      let e = tid + r * 256u;
      { ${aIdx}
        let gm = row0 + m; let gk = k0 + k;
        var v = 0.0;
        if (gm < p.M && gk < p.K) { v = ${aAt}; }
        As[k * 64u + m] = v; }
      { ${bIdx}
        let gn = col0 + n; let gk = k0 + k;
        var v = 0.0;
        if (gn < p.N && gk < p.K) { v = ${bAt}; }
        Bs[k * 64u + n] = v; }
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[k * 64u + lid.y + i * 16u]; }
      for (var j = 0u; j < 4u; j++) { b[j] = Bs[k * 64u + lid.x + j * 16u]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]); }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = row0 + lid.y + i * 16u;
    if (m >= p.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = col0 + lid.x + j * 16u;
      if (n >= p.N) { continue; }
      let o = m * p.N + n;
      if (p.beta == 0.0) { C[o] = acc[i * 4u + j]; } else { C[o] = C[o] + acc[i * 4u + j]; }
    }
  }
}`;
}
//#endregion

const SHADERS = {
  embed: /* wgsl */ `
struct P { n: u32, d: u32 }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> E: array<f32>;
@group(0) @binding(2) var<storage, read> tok: array<i32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n * p.d) { return; }
  let row = i / p.d;
  out[i] = E[u32(tok[row]) * p.d + i % p.d];
}`,

  // Embedding rückwärts ohne Float-Atomics: Die CPU sortiert die Positionen
  // nach Token-ID (CSR), jeder Thread summiert für ein (Token, Dimension).
  embedBwd: /* wgsl */ `
struct P { nUnique: u32, d: u32 }
@group(0) @binding(0) var<storage, read_write> dE: array<f32>;
@group(0) @binding(1) var<storage, read> dx: array<f32>;
@group(0) @binding(2) var<storage, read> uniq: array<i32>;
@group(0) @binding(3) var<storage, read> starts: array<i32>;
@group(0) @binding(4) var<storage, read> pos: array<i32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.nUnique * p.d) { return; }
  let u = i / p.d; let j = i % p.d;
  var s = 0.0;
  for (var q = starts[u]; q < starts[u + 1u]; q++) { s += dx[u32(pos[q]) * p.d + j]; }
  let o = u32(uniq[u]) * p.d + j;
  dE[o] = dE[o] + s;
}`,

  rmsnorm: /* wgsl */ `
struct P { d: u32, eps: f32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let o = wg.x * p.d;
  var s = 0.0;
  for (var j = lid.x; j < p.d; j += 64u) { s += x[o + j] * x[o + j]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  let r = 1.0 / sqrt(red[0] / f32(p.d) + p.eps);
  if (lid.x == 0u) { rstd[wg.x] = r; }
  for (var j = lid.x; j < p.d; j += 64u) { y[o + j] = g[j] * x[o + j] * r; }
}`,

  rmsnormBwdX: /* wgsl */ `
struct P { d: u32 }
@group(0) @binding(0) var<storage, read_write> dx: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let o = wg.x * p.d;
  var s = 0.0;
  for (var j = lid.x; j < p.d; j += 64u) { s += g[j] * dy[o + j] * x[o + j]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  let r = rstd[wg.x];
  let c = r * r * r * red[0] / f32(p.d);
  for (var j = lid.x; j < p.d; j += 64u) { dx[o + j] = dx[o + j] + r * g[j] * dy[o + j] - c * x[o + j]; }
}`,

  rmsnormBwdG: /* wgsl */ `
struct P { n: u32, d: u32 }
@group(0) @binding(0) var<storage, read_write> dg: array<f32>;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let j = wg.x;
  var s = 0.0;
  for (var n = lid.x; n < p.n; n += 64u) { s += dy[n * p.d + j] * x[n * p.d + j] * rstd[n]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  if (lid.x == 0u) { dg[j] = red[0]; }
}`,

  rope: /* wgsl */ `
struct P { n: u32, T: u32, heads: u32, hd: u32, base: f32, sign: f32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  let half = p.hd / 2u;
  if (i >= p.n * p.heads * half) { return; }
  let k = i % half;
  let row = i / (p.heads * half);
  let h = (i / half) % p.heads;
  let theta = f32(row % p.T) * pow(p.base, -2.0 * f32(k) / f32(p.hd));
  let c = cos(theta); let s = p.sign * sin(theta);
  let o = (row * p.heads + h) * p.hd + 2u * k;
  let a = x[o]; let b = x[o + 1u];
  x[o] = a * c - b * s;
  x[o + 1u] = a * s + b * c;
}`,

  // Eine Workgroup pro (Batch, Kopf, Query-Position i).
  attention: /* wgsl */ `
struct P { B: u32, T: u32, nH: u32, nKV: u32, hd: u32 }
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read_write> P_: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> k: array<f32>;
@group(0) @binding(4) var<storage, read> v: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
var<workgroup> qs: array<f32, 256>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.B * p.nH * p.T) { return; }
  let i = row % p.T;
  let h = (row / p.T) % p.nH;
  let b = row / (p.T * p.nH);
  let kh = h / (p.nH / p.nKV);
  let qstride = p.nH * p.hd; let kstride = p.nKV * p.hd;
  let qi = (b * p.T + i) * qstride + h * p.hd;
  let pRow = row * p.T;
  let scale = 1.0 / sqrt(f32(p.hd));
  for (var t = lid.x; t < p.hd; t += 64u) { qs[t] = q[qi + t]; }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = lid.x; j <= i; j += 64u) {
    let kj = (b * p.T + j) * kstride + kh * p.hd;
    var s = 0.0;
    for (var t = 0u; t < p.hd; t++) { s = fma(qs[t], k[kj + t], s); }
    s *= scale;
    P_[pRow + j] = s;
    mx = max(mx, s);
  }
  red[lid.x] = mx;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] = max(red[lid.x], red[lid.x + w]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j <= i; j += 64u) { let e = exp(P_[pRow + j] - m); P_[pRow + j] = e; sum += e; }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var j = lid.x; j < p.T; j += 64u) { if (j <= i) { P_[pRow + j] = P_[pRow + j] * inv; } else { P_[pRow + j] = 0.0; } }
  storageBarrier();
  workgroupBarrier();
  for (var t = lid.x; t < p.hd; t += 64u) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc = fma(P_[pRow + j], v[(b * p.T + j) * kstride + kh * p.hd + t], acc); }
    o[qi + t] = acc;
  }
}`,

  // Rückwärts Teil 1, pro Query-Zeile: dP, dS (= P ⊙ (dP − Σ P·dP) · scale) und dQ.
  attentionBwdQ: /* wgsl */ `
struct P { B: u32, T: u32, nH: u32, nKV: u32, hd: u32 }
@group(0) @binding(0) var<storage, read_write> dq: array<f32>;
@group(0) @binding(1) var<storage, read_write> dS: array<f32>;
@group(0) @binding(2) var<storage, read> dO: array<f32>;
@group(0) @binding(3) var<storage, read> P_: array<f32>;
@group(0) @binding(4) var<storage, read> k: array<f32>;
@group(0) @binding(5) var<storage, read> v: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
var<workgroup> dos: array<f32, 256>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.B * p.nH * p.T) { return; }
  let i = row % p.T;
  let h = (row / p.T) % p.nH;
  let b = row / (p.T * p.nH);
  let kh = h / (p.nH / p.nKV);
  let qstride = p.nH * p.hd; let kstride = p.nKV * p.hd;
  let qi = (b * p.T + i) * qstride + h * p.hd;
  let pRow = row * p.T;
  let scale = 1.0 / sqrt(f32(p.hd));
  for (var t = lid.x; t < p.hd; t += 64u) { dos[t] = dO[qi + t]; }
  workgroupBarrier();
  var s = 0.0;
  for (var j = lid.x; j <= i; j += 64u) {
    let vj = (b * p.T + j) * kstride + kh * p.hd;
    var dp = 0.0;
    for (var t = 0u; t < p.hd; t++) { dp = fma(dos[t], v[vj + t], dp); }
    dS[pRow + j] = dp;
    s += dp * P_[pRow + j];
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  let total = red[0];
  for (var j = lid.x; j < p.T; j += 64u) {
    if (j <= i) { dS[pRow + j] = P_[pRow + j] * (dS[pRow + j] - total) * scale; } else { dS[pRow + j] = 0.0; }
  }
  storageBarrier();
  workgroupBarrier();
  for (var t = lid.x; t < p.hd; t += 64u) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc = fma(dS[pRow + j], k[(b * p.T + j) * kstride + kh * p.hd + t], acc); }
    dq[qi + t] = acc;
  }
}`,

  // Rückwärts Teil 2, pro Key-Position j und KV-Kopf: dK und dV, summiert über
  // alle Query-Köpfe der Gruppe und alle späteren Positionen i ≥ j.
  attentionBwdKV: /* wgsl */ `
struct P { B: u32, T: u32, nH: u32, nKV: u32, hd: u32 }
@group(0) @binding(0) var<storage, read_write> dk: array<f32>;
@group(0) @binding(1) var<storage, read_write> dv: array<f32>;
@group(0) @binding(2) var<storage, read> dS: array<f32>;
@group(0) @binding(3) var<storage, read> P_: array<f32>;
@group(0) @binding(4) var<storage, read> q: array<f32>;
@group(0) @binding(5) var<storage, read> dO: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let col = wg.x + wg.y * 65535u;
  if (col >= p.B * p.nKV * p.T) { return; }
  let j = col % p.T;
  let kh = (col / p.T) % p.nKV;
  let b = col / (p.T * p.nKV);
  let group = p.nH / p.nKV;
  let qstride = p.nH * p.hd; let kstride = p.nKV * p.hd;
  for (var t = lid.x; t < p.hd; t += 64u) {
    var ak = 0.0; var av = 0.0;
    for (var g = 0u; g < group; g++) {
      let h = kh * group + g;
      for (var i = j; i < p.T; i++) {
        let pIdx = ((b * p.nH + h) * p.T + i) * p.T + j;
        let qi = (b * p.T + i) * qstride + h * p.hd + t;
        ak = fma(dS[pIdx], q[qi], ak);
        av = fma(P_[pIdx], dO[qi], av);
      }
    }
    let kj = (b * p.T + j) * kstride + kh * p.hd + t;
    dk[kj] = ak;
    dv[kj] = av;
  }
}`,

  swiglu: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  let s = 1.0 / (1.0 + exp(-g[i]));
  h[i] = g[i] * s * u[i];
}`,

  swigluBwd: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> dg: array<f32>;
@group(0) @binding(1) var<storage, read_write> du: array<f32>;
@group(0) @binding(2) var<storage, read> dh: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read> u: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  let s = 1.0 / (1.0 + exp(-g[i]));
  du[i] = dh[i] * g[i] * s;
  dg[i] = dh[i] * u[i] * s * (1.0 + g[i] * (1.0 - s));
}`,

  add: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  x[i] = x[i] + y[i];
}`,

  copy: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  x[i] = y[i];
}`,

  zero: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  x[i] = 0.0;
}`,

  // Eine Workgroup pro Zeile: Maximum, Summe, Loss der Zeile, dLogits.
  crossEntropy: /* wgsl */ `
struct P { N: u32, V: u32 }
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> rowLoss: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<i32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wg.x;
  let o = n * p.V;
  var mx = -3.0e38;
  for (var j = lid.x; j < p.V; j += 256u) { mx = max(mx, logits[o + j]); }
  red[lid.x] = mx;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] = max(red[lid.x], red[lid.x + w]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = lid.x; j < p.V; j += 256u) { s += exp(logits[o + j] - m); }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  let sum = red[0];
  let t = u32(targets[n]);
  if (lid.x == 0u) { rowLoss[n] = log(sum) - (logits[o + t] - m); }
  storageBarrier();
  workgroupBarrier();
  let invN = 1.0 / f32(p.N);
  for (var j = lid.x; j < p.V; j += 256u) {
    var gj = exp(logits[o + j] - m) / sum * invN;
    if (j == t) { gj -= invN; }
    logits[o + j] = gj;
  }
}`,

  // Summe eines Arrays in zwei Stufen: viele Teilsummen, dann eine Gesamtsumme.
  sumPartial: /* wgsl */ `
struct P { n: u32, square: u32, scale: f32 }
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let start = wg.x * 4096u;
  var s = 0.0;
  for (var r = 0u; r < 16u; r++) {
    let i = start + r * 256u + lid.x;
    if (i < p.n) { let v = x[i]; if (p.square == 1u) { s += v * v; } else { s += v; } }
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) { if (lid.x < w) { red[lid.x] += red[lid.x + w]; } workgroupBarrier(); }
  if (lid.x == 0u) { out[wg.x] = red[0] * p.scale; }
}`,

  adamw: /* wgsl */ `
struct P { n: u32, wd: f32 }
struct S { lr: f32, beta1: f32, beta2: f32, eps: f32, bc1: f32, bc2: f32, maxNorm: f32, pad: f32 }
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> normSq: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<uniform> s: S;
@compute @workgroup_size(${WG}) fn main(${ELEM_ARGS}) {
  ${IDX}
  if (i >= p.n) { return; }
  let clip = min(1.0, s.maxNorm / (sqrt(normSq[0]) + 1e-6));
  let g = grad[i] * clip;
  m[i] = s.beta1 * m[i] + (1.0 - s.beta1) * g;
  v[i] = s.beta2 * v[i] + (1.0 - s.beta2) * g * g;
  param[i] = param[i] - s.lr * ((m[i] / s.bc1) / (sqrt(v[i] / s.bc2) + s.eps) + p.wd * param[i]);
}`,
};

export async function createWebGpuBackend() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage),
    },
  });
  let lost = null;
  device.lost.then((info) => { lost = info; });

  const pipelines = {};
  const pipeline = (name) => {
    if (!pipelines[name]) {
      const code = name.startsWith('matmul') ? matmulShader(name[6] === 'T', name[7] === 'T') : SHADERS[name];
      pipelines[name] = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code, label: name }), entryPoint: 'main' } });
    }
    return pipelines[name];
  };

  let nextId = 1;
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const alloc = (n, int = false) => {
    const size = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    return { buffer: device.createBuffer({ size, usage: STORAGE }), id: nextId++, offset: 0, size: n, int, shadow: null };
  };
  const view = (h, offset, size) => ({ ...h, offset: h.offset + offset, size, id: h.id });

  // Uniform-Buffer und Bind-Groups werden pro Aufruf-Signatur einmal angelegt
  // und dann jeden Schritt wiederverwendet.
  const cache = new Map();
  const uniformBuffer = (data) => {
    const buf = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  };
  const packUniforms = (fields) => {
    const buf = new ArrayBuffer(fields.length * 4);
    const dv = new DataView(buf);
    fields.forEach(([type, v], i) => (type === 'u' ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v, true)));
    return new Uint8Array(buf);
  };

  let encoder = null;
  let pass = null;
  const begin = () => {
    if (!encoder) encoder = device.createCommandEncoder();
    if (!pass) pass = encoder.beginComputePass();
  };
  const endPass = () => { if (pass) { pass.end(); pass = null; } };
  const submit = () => {
    endPass();
    if (encoder) { device.queue.submit([encoder.finish()]); encoder = null; }
  };

  // Eine Shader-Ausführung einreihen. buffers: Handles in Binding-Reihenfolge,
  // uniforms: [['u'|'f', Wert], …], extra: zusätzlicher fester Uniform-Buffer.
  function run(name, buffers, uniforms, groups, extra = null) {
    if (lost) throw new Error(`GPU verloren: ${lost.message}`);
    const key = `${name}|${buffers.map((b) => `${b.id}:${b.offset}:${b.size}`).join(',')}|${uniforms.map((u) => u[1]).join(',')}`;
    let bg = cache.get(key);
    if (!bg) {
      const pl = pipeline(name);
      const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b.buffer, offset: b.offset * 4, size: Math.max(4, b.size * 4) } }));
      entries.push({ binding: buffers.length, resource: { buffer: uniformBuffer(packUniforms(uniforms)) } });
      if (extra) entries.push({ binding: buffers.length + 1, resource: { buffer: extra } });
      bg = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries });
      cache.set(key, bg);
    }
    begin();
    pass.setPipeline(pipeline(name));
    pass.setBindGroup(0, bg);
    const [x, y] = typeof groups === 'number' ? split(groups) : groups;
    pass.dispatchWorkgroups(x, y);
  }
  const split = (n) => (n <= 65535 ? [Math.max(1, n), 1] : [65535, Math.ceil(n / 65535)]);
  const elemGroups = (n) => {
    const wgs = Math.ceil(n / WG);
    if (wgs <= 65535) return [Math.max(1, wgs), 1];
    const x = 65535;
    return [x, Math.ceil(wgs / x)];
  };

  // Hilfspuffer für Reduktionen und den Embedding-Rückwärtslauf
  const scratch = {};
  const scratchBuf = (name, n, int = false) => {
    if (!scratch[name] || scratch[name].size < n) scratch[name] = alloc(n, int);
    return scratch[name];
  };
  const stepUniforms = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let stepKey = '';

  // Summe (bzw. Quadratsumme) von x in out[0], in Stufen zu je 4096 Werten.
  function sumInto(out, x, n, square, scale = 1) {
    let count = Math.ceil(n / 4096);
    if (count === 1) { run('sumPartial', [out, x], [['u', n], ['u', square ? 1 : 0], ['f', scale]], 1); return; }
    let cur = scratchBuf(`partial${count}`, count);
    run('sumPartial', [cur, x], [['u', n], ['u', square ? 1 : 0], ['f', 1]], count);
    while (count > 4096) {
      const next = Math.ceil(count / 4096);
      const target = scratchBuf(`partial${next}`, next);
      run('sumPartial', [target, cur], [['u', count], ['u', 0], ['f', 1]], next);
      cur = target;
      count = next;
    }
    run('sumPartial', [out, { ...cur, size: count }], [['u', count], ['u', 0], ['f', scale]], 1);
  }

  const be = {
    kind: 'webgpu',
    adapterInfo: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } : {},
    alloc: (n) => alloc(n),
    allocInt: (n) => alloc(n, true),
    view,
    write(h, data, offset = 0) {
      const arr = h.int ? Int32Array.from(data) : (data instanceof Float32Array ? data : Float32Array.from(data));
      if (h.int) h.shadow = arr;
      device.queue.writeBuffer(h.buffer, (h.offset + offset) * 4, arr.buffer, arr.byteOffset, arr.byteLength);
    },
    async read(h) {
      submit();
      const bytes = Math.max(4, h.size * 4);
      const staging = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(h.buffer, h.offset * 4, staging, 0, Math.ceil(bytes / 4) * 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0, h.size * 4));
      staging.destroy();
      return out;
    },
    async flush() { submit(); },

    matmul(C, A, B, M, N, K, transA, transB, beta = 0) {
      run(`matmul${transA ? 'T' : 'N'}${transB ? 'T' : 'N'}`, [A, B, C], [['u', M], ['u', N], ['u', K], ['f', beta]], [Math.ceil(N / 64), Math.ceil(M / 64)]);
    },
    embed(out, E, tokens, N, d) { run('embed', [out, E, tokens], [['u', N], ['u', d]], elemGroups(N * d)); },
    embedBwd(dE, dx, tokens, N, d) {
      // CSR auf der CPU: welche Positionen gehören zu welchem Token?
      const ids = tokens.shadow;
      const byToken = new Map();
      for (let n = 0; n < N; n++) { const t = ids[n]; if (!byToken.has(t)) byToken.set(t, []); byToken.get(t).push(n); }
      const uniq = [...byToken.keys()];
      const starts = [0];
      const pos = [];
      for (const t of uniq) { pos.push(...byToken.get(t)); starts.push(pos.length); }
      const key = `csr${tokens.id}`;
      const U = scratchBuf(`${key}u`, N + 1, true), S = scratchBuf(`${key}s`, N + 2, true), Pp = scratchBuf(`${key}p`, N, true);
      device.queue.writeBuffer(U.buffer, 0, Int32Array.from(uniq));
      device.queue.writeBuffer(S.buffer, 0, Int32Array.from(starts));
      device.queue.writeBuffer(Pp.buffer, 0, Int32Array.from(pos));
      run('embedBwd', [dE, dx, { ...U, size: N + 1 }, { ...S, size: N + 2 }, { ...Pp, size: N }], [['u', uniq.length], ['u', d]], elemGroups(uniq.length * d));
    },
    rmsnorm(y, rstd, x, g, N, d, eps) { run('rmsnorm', [y, rstd, x, g], [['u', d], ['f', eps]], N); },
    rmsnormBwd(dx, dg, dy, x, g, rstd, N, d) {
      run('rmsnormBwdG', [dg, dy, x, rstd], [['u', N], ['u', d]], d);
      run('rmsnormBwdX', [dx, dy, x, g, rstd], [['u', d]], N);
    },
    rope(x, N, T, heads, hd, base, inverse = false) {
      run('rope', [x], [['u', N], ['u', T], ['u', heads], ['u', hd], ['f', base], ['f', inverse ? -1 : 1]], elemGroups((N * heads * hd) / 2));
    },
    attention(o, P, q, k, v, B, T, nH, nKV, hd) {
      run('attention', [o, P, q, k, v], [['u', B], ['u', T], ['u', nH], ['u', nKV], ['u', hd]], split(B * nH * T));
    },
    attentionBwd(dq, dk, dv, dS, dO, P, q, k, v, B, T, nH, nKV, hd) {
      const u = [['u', B], ['u', T], ['u', nH], ['u', nKV], ['u', hd]];
      run('attentionBwdQ', [dq, dS, dO, P, k, v], u, split(B * nH * T));
      run('attentionBwdKV', [dk, dv, dS, P, q, dO], u, split(B * nKV * T));
    },
    swiglu(h, g, u, n) { run('swiglu', [h, g, u], [['u', n]], elemGroups(n)); },
    swigluBwd(dg, du, dh, g, u, n) { run('swigluBwd', [dg, du, dh, g, u], [['u', n]], elemGroups(n)); },
    add(x, y, n) { run('add', [x, y], [['u', n]], elemGroups(n)); },
    copy(x, y, n) { run('copy', [x, y], [['u', n]], elemGroups(n)); },
    zero(x, n) { run('zero', [x], [['u', n]], elemGroups(n)); },
    crossEntropy(logits, targets, N, V, lossOut) {
      const rows = scratchBuf('rowLoss', N);
      run('crossEntropy', [logits, rows, targets], [['u', N], ['u', V]], N);
      sumInto(lossOut, { ...rows, size: N }, N, false, 1 / N);
    },
    sumSquares(x, n, out) { sumInto(out, x, n, true); },
    adamw(p, g, m, v, n, hp, normSq) {
      const key = `${hp.lr}|${hp.t}|${hp.maxNorm}`;
      if (key !== stepKey) {
        stepKey = key;
        device.queue.writeBuffer(stepUniforms, 0, new Float32Array([hp.lr, hp.beta1, hp.beta2, hp.eps, 1 - hp.beta1 ** hp.t, 1 - hp.beta2 ** hp.t, hp.maxNorm, 0]));
      }
      run('adamw', [p, g, m, v, normSq], [['u', n], ['f', hp.wd]], elemGroups(n), stepUniforms);
    },
    destroy() { device.destroy(); },
  };
  return be;
}
