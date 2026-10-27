// =====================================================================
// backend-cpu.js – Das Rechenwerk in reinem JavaScript
// =====================================================================
//
// In Teil 1 war jede Zahl ein Objekt mit eigenem Gradienten. Das ist
// anschaulich, aber 100× zu langsam. Echte Frameworks rechnen mit
// Tensoren: lange Zahlenreihen (Float32Array), auf denen ganze Operationen
// auf einmal laufen. Für jede Operation schreibt man die Vorwärts- UND die
// Rückwärtsrechnung von Hand – genau das passiert hier.
//
// Dieselben Operationen gibt es noch einmal in backend-webgpu.js für die
// Grafikkarte. Diese Datei ist die Referenz, gegen die die GPU geprüft wird.

export function createCpuBackend(ArrayType = Float32Array) {
  const view = (buf, offset, size) => buf.subarray(offset, offset + size);

  //#region cpu-matmul
  // C (M×N) = op(A) · op(B), op = optional transponiert.
  //   beta = 0: C wird überschrieben, beta = 1: aufaddiert.
  // Die Schleifen sind so verschachtelt, dass der Speicher der Reihe nach
  // gelesen wird – das macht in JavaScript locker Faktor 5 aus.
  function matmul(C, A, B, M, N, K, transA, transB, beta = 0) {
    if (beta === 0) C.fill(0, 0, M * N);
    if (!transA && transB) {                 // C = A · Bᵀ   (Linear vorwärts)
      for (let i = 0; i < M; i++) {
        const ai = i * K;
        for (let j = 0; j < N; j++) {
          const bj = j * K;
          let s = 0;
          for (let k = 0; k < K; k++) s += A[ai + k] * B[bj + k];
          C[i * N + j] += s;
        }
      }
    } else if (!transA && !transB) {         // C = A · B    (Gradient der Eingabe)
      for (let i = 0; i < M; i++) {
        const ci = i * N;
        for (let k = 0; k < K; k++) {
          const a = A[i * K + k];
          if (a === 0) continue;
          const bk = k * N;
          for (let j = 0; j < N; j++) C[ci + j] += a * B[bk + j];
        }
      }
    } else if (transA && !transB) {          // C = Aᵀ · B   (Gradient der Gewichte)
      for (let k = 0; k < K; k++) {
        const ak = k * M, bk = k * N;
        for (let i = 0; i < M; i++) {
          const a = A[ak + i];
          if (a === 0) continue;
          const ci = i * N;
          for (let j = 0; j < N; j++) C[ci + j] += a * B[bk + j];
        }
      }
    } else {
      throw new Error('matmul: Aᵀ·Bᵀ wird nicht gebraucht');
    }
  }
  //#endregion

  function embed(out, E, tokens, N, d) {
    for (let n = 0; n < N; n++) out.set(E.subarray(tokens[n] * d, tokens[n] * d + d), n * d);
  }

  // dE[token] += dx[n] für jede Position n.
  function embedBwd(dE, dx, tokens, N, d) {
    for (let n = 0; n < N; n++) {
      const e = tokens[n] * d, x = n * d;
      for (let j = 0; j < d; j++) dE[e + j] += dx[x + j];
    }
  }

  //#region cpu-rmsnorm
  // RMSNorm: y = g ⊙ x / rms(x),  rms(x) = sqrt(mean(x²) + ε)
  function rmsnorm(y, rstd, x, g, N, d, eps) {
    for (let n = 0; n < N; n++) {
      const o = n * d;
      let ss = 0;
      for (let j = 0; j < d; j++) ss += x[o + j] * x[o + j];
      const r = 1 / Math.sqrt(ss / d + eps);
      rstd[n] = r;
      for (let j = 0; j < d; j++) y[o + j] = g[j] * x[o + j] * r;
    }
  }

  // Rückwärts:  dx += r·(g⊙dy) − r³/d · x · Σ(g⊙dy⊙x),   dg = Σ_n dy ⊙ x · r
  function rmsnormBwd(dx, dg, dy, x, g, rstd, N, d) {
    dg.fill(0, 0, d);
    for (let n = 0; n < N; n++) {
      const o = n * d, r = rstd[n];
      let dot = 0;
      for (let j = 0; j < d; j++) dot += g[j] * dy[o + j] * x[o + j];
      const c = (r * r * r * dot) / d;
      for (let j = 0; j < d; j++) {
        dx[o + j] += r * g[j] * dy[o + j] - c * x[o + j];
        dg[j] += dy[o + j] * x[o + j] * r;
      }
    }
  }
  //#endregion

  //#region cpu-rope
  // RoPE: Jedes Zahlenpaar (x₂ᵢ, x₂ᵢ₊₁) eines Kopfes wird um den Winkel
  // pos · base^(−2i/hd) gedreht. Nahe Positionen drehen ähnlich weit, so
  // "spürt" die Attention den Abstand zwischen zwei Tokens.
  // Rückwärts ist einfach die Drehung in die Gegenrichtung (inverse = true).
  function rope(x, N, T, nHeads, hd, base, inverse = false) {
    const half = hd / 2;
    for (let n = 0; n < N; n++) {
      const pos = n % T;
      for (let h = 0; h < nHeads; h++) {
        const o = (n * nHeads + h) * hd;
        for (let i = 0; i < half; i++) {
          const theta = pos * Math.pow(base, (-2 * i) / hd);
          const c = Math.cos(theta), s = inverse ? -Math.sin(theta) : Math.sin(theta);
          const a = x[o + 2 * i], b = x[o + 2 * i + 1];
          x[o + 2 * i] = a * c - b * s;
          x[o + 2 * i + 1] = a * s + b * c;
        }
      }
    }
  }
  //#endregion

  //#region cpu-attention
  // Kausale Multi-Head-Attention mit Grouped-Query-Attention (GQA):
  // Mehrere Query-Köpfe teilen sich einen Key/Value-Kopf (spart Speicher
  // beim Generieren – so machen es Llama 3, Mistral, Qwen …).
  function attention(o, P, q, k, v, B, T, nH, nKV, hd) {
    const scale = 1 / Math.sqrt(hd), qs = nH * hd, ks = nKV * hd, group = nH / nKV;
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < nH; h++) {
        const kh = Math.floor(h / group);
        for (let i = 0; i < T; i++) {
          const pRow = ((b * nH + h) * T + i) * T;
          const qi = (b * T + i) * qs + h * hd;
          let max = -Infinity;
          for (let j = 0; j <= i; j++) {
            const kj = (b * T + j) * ks + kh * hd;
            let s = 0;
            for (let t = 0; t < hd; t++) s += q[qi + t] * k[kj + t];
            s *= scale;
            P[pRow + j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= i; j++) { const e = Math.exp(P[pRow + j] - max); P[pRow + j] = e; sum += e; }
          for (let j = 0; j <= i; j++) P[pRow + j] /= sum;
          for (let j = i + 1; j < T; j++) P[pRow + j] = 0;   // Zukunft: maskiert
          for (let t = 0; t < hd; t++) {
            let acc = 0;
            for (let j = 0; j <= i; j++) acc += P[pRow + j] * v[(b * T + j) * ks + kh * hd + t];
            o[qi + t] = acc;
          }
        }
      }
    }
  }

  // Rückwärts durch Softmax und die beiden Matrixprodukte:
  //   dP = dO·Vᵀ,  dS = P ⊙ (dP − Σ P⊙dP),  dQ = dS·K·scale,  dK = dSᵀ·Q·scale,  dV = Pᵀ·dO
  function attentionBwd(dq, dk, dv, dS, dO, P, q, k, v, B, T, nH, nKV, hd) {
    const scale = 1 / Math.sqrt(hd), qs = nH * hd, ks = nKV * hd, group = nH / nKV;
    dk.fill(0, 0, B * T * ks);
    dv.fill(0, 0, B * T * ks);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < nH; h++) {
        const kh = Math.floor(h / group);
        for (let i = 0; i < T; i++) {
          const pRow = ((b * nH + h) * T + i) * T;
          const qi = (b * T + i) * qs + h * hd;
          let sum = 0;
          for (let j = 0; j <= i; j++) {
            const vj = (b * T + j) * ks + kh * hd;
            let dp = 0;
            for (let t = 0; t < hd; t++) dp += dO[qi + t] * v[vj + t];
            dS[pRow + j] = dp;
            sum += dp * P[pRow + j];
          }
          for (let t = 0; t < hd; t++) dq[qi + t] = 0;
          for (let j = 0; j <= i; j++) {
            const p = P[pRow + j];
            const ds = p * (dS[pRow + j] - sum) * scale;
            const kj = (b * T + j) * ks + kh * hd;
            for (let t = 0; t < hd; t++) {
              dq[qi + t] += ds * k[kj + t];
              dk[kj + t] += ds * q[qi + t];
              dv[kj + t] += p * dO[qi + t];
            }
          }
        }
      }
    }
  }
  //#endregion

  //#region cpu-swiglu
  // SwiGLU: h = silu(g) ⊙ u,  silu(g) = g · σ(g)
  // Das "Tor" g entscheidet, wie viel von u durchgelassen wird.
  function swiglu(h, g, u, n) {
    for (let i = 0; i < n; i++) {
      const s = 1 / (1 + Math.exp(-g[i]));
      h[i] = g[i] * s * u[i];
    }
  }
  function swigluBwd(dg, du, dh, g, u, n) {
    for (let i = 0; i < n; i++) {
      const s = 1 / (1 + Math.exp(-g[i]));
      const silu = g[i] * s;
      du[i] = dh[i] * silu;
      dg[i] = dh[i] * u[i] * s * (1 + g[i] * (1 - s));
    }
  }
  //#endregion

  function add(x, y, n) { for (let i = 0; i < n; i++) x[i] += y[i]; }
  function copy(dst, src, n) { dst.set(src.subarray(0, n)); }

  //#region cpu-cross-entropy
  // Softmax + Cross-Entropy + Gradient in einem Durchgang. Die Logits
  // werden dabei durch ihren Gradienten (p − onehot) / N ersetzt.
  function crossEntropy(logits, targets, N, V, lossOut) {
    let total = 0;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      let max = -Infinity;
      for (let j = 0; j < V; j++) if (logits[o + j] > max) max = logits[o + j];
      const target = logits[o + targets[n]] - max;
      let sum = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(logits[o + j] - max); logits[o + j] = e; sum += e; }
      total += Math.log(sum) - target;              // −log p(Ziel)
      for (let j = 0; j < V; j++) logits[o + j] = logits[o + j] / sum / N;
      logits[o + targets[n]] -= 1 / N;
    }
    lossOut[0] = total / N;
  }
  //#endregion

  function sumSquares(x, n, out) { let s = 0; for (let i = 0; i < n; i++) s += x[i] * x[i]; out[0] = s; }
  function zero(x, n) { x.fill(0, 0, n); }

  //#region cpu-adamw
  // AdamW mit "entkoppeltem" Weight Decay: Gewichte schrumpfen leicht in
  // Richtung 0, unabhängig vom Gradienten. clip skaliert alle Gradienten,
  // wenn ihre Gesamtlänge zu groß ist (Gradient Clipping).
  function adamw(p, g, m, v, n, { lr, beta1, beta2, eps, wd, t, maxNorm }, normSq) {
    const bc1 = 1 - beta1 ** t, bc2 = 1 - beta2 ** t;
    const clip = Math.min(1, maxNorm / (Math.sqrt(normSq[0]) + 1e-6));
    for (let i = 0; i < n; i++) {
      const gi = g[i] * clip;
      m[i] = beta1 * m[i] + (1 - beta1) * gi;
      v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
      p[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps) + wd * p[i]);
    }
  }
  //#endregion

  return {
    kind: 'cpu',
    alloc: (n) => new ArrayType(n),
    allocInt: (n) => new Int32Array(n),
    view,
    write: (buf, data, offset = 0) => buf.set(data, offset),
    read: async (buf) => buf.slice(),
    matmul, embed, embedBwd, rmsnorm, rmsnormBwd, rope, attention, attentionBwd,
    swiglu, swigluBwd, add, copy, zero, crossEntropy, sumSquares, adamw,
    flush: async () => {},
  };
}
