// Client-side CBCES and NCD approximation demo
(function () {
  const enc = new TextEncoder();

  function toBytes(str) {
    return enc.encode(str || "");
  }

  function counts256(bytes) {
    const c = new Uint32Array(256);
    for (let i = 0; i < bytes.length; i++) c[bytes[i]]++;
    return c;
  }

  function addCounts(a, b) {
    const out = new Uint32Array(256);
    for (let i = 0; i < 256; i++) out[i] = a[i] + b[i];
    return out;
  }

  function sum(arr) {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s;
  }

  function entropyBits(counts) {
    const N = sum(counts);
    if (!N) return 0;
    let H = 0;
    for (let i = 0; i < 256; i++) {
      const p = counts[i] / N;
      if (p > 0) H -= p * Math.log2(p);
    }
    return H; // bits/byte
  }

  // Cross-entropy of candidate vs baseline with Laplace smoothing (alpha=1)
  function crossEntropyBits(baseCounts, candCounts) {
    const alpha = 1;
    const K = 256;
    const Nbase = sum(baseCounts);
    const Ncand = sum(candCounts);
    if (!Ncand) return 0;
    let Hpq = 0;
    for (let i = 0; i < 256; i++) {
      const qc = candCounts[i];
      if (!qc) continue;
      const p = (baseCounts[i] + alpha) / (Nbase + alpha * K);
      Hpq += (qc / Ncand) * (-Math.log2(p));
    }
    return Hpq;
  }

  // NCD approximation using entropy-coded size: size(X) ≈ |X| * H(X)
  function ncdApprox(baseCounts, candCounts) {
    const Na = sum(baseCounts);
    const Nb = sum(candCounts);
    if (!Na || !Nb) return 0;
    const Ha = entropyBits(baseCounts);
    const Hb = entropyBits(candCounts);
    const Hab = entropyBits(addCounts(baseCounts, candCounts));
    const Ca = Na * Ha;
    const Cb = Nb * Hb;
    const Cab = (Na + Nb) * Hab;
    const num = Cab - Math.min(Ca, Cb);
    const den = Math.max(Ca, Cb) || 1;
    return num / den;
  }

  function preset(scenario) {
    if (scenario === "biomedical") {
      return {
        baseline: [
          "GENE001, 3.2, 2.9, 3.1, 2.8, 3.0, 3.1",
          "GENE014, 1.1, 1.0, 1.0, 1.2, 1.1, 1.0",
          "GENE087, 0.8, 0.9, 0.8, 0.7, 0.9, 0.8",
          "GENE203, 2.2, 2.0, 2.1, 2.2, 2.1, 2.0",
          "GENE119, 1.5, 1.6, 1.5, 1.6, 1.5, 1.5",
        ].join("\n"),
        candidate:
          "GENE014, 5.9, 6.1, 5.8, 6.0, 6.2, 5.9  # sudden spike (anomaly)",
      };
    }
    // default: security
    return {
      baseline: [
        "GET /api/v1/users 200 122ms Mozilla/5.0",
        "GET /api/v1/users 200 118ms Mozilla/5.0",
        "POST /api/v1/login 302 96ms Mozilla/5.0",
        "GET /health 200 3ms curl/7.79.1",
        "GET /api/v1/users/42 200 133ms Mozilla/5.0",
      ].join("\n"),
      candidate:
        "GET /admin/delete?user=42;DROP%20TABLE%20users 200 2ms sqlmap/1.6.10",
    };
  }

  function parseBaseline(text) {
    const lines = (text || "").split(/\r?\n/).filter((l) => l.trim().length);
    return toBytes(lines.join("\n"));
  }

  function fmt(x) {
    if (!isFinite(x)) return "–";
    return x.toFixed(3);
  }

  function run() {
    const baseBytes = parseBaseline(document.getElementById("baseline").value);
    const candBytes = toBytes(document.getElementById("candidate").value);
    const th = parseFloat(document.getElementById("threshold").value);

    const cBase = counts256(baseBytes);
    const cCand = counts256(candBytes);
    const Hbase = entropyBits(cBase);
    const Hx = crossEntropyBits(cBase, cCand);
    const delta = Hx - Hbase;
    const ncd = ncdApprox(cBase, cCand);

    document.getElementById("hBase").textContent = fmt(Hbase);
    document.getElementById("hX").textContent = fmt(Hx);
    document.getElementById("delta").textContent = fmt(delta);
    document.getElementById("ncd").textContent = fmt(ncd);

    const badge = document.getElementById("badge");
    const status = document.getElementById("status");
    badge.innerHTML = "";
    if (delta > th) {
      status.textContent = "Likely Anomaly";
      const el = document.createElement("span");
      el.className = "badge badge-bad";
      el.textContent = `Δ>${fmt(th)}`;
      badge.appendChild(el);
    } else {
      status.textContent = "Looks Normal";
      const el = document.createElement("span");
      el.className = "badge badge-good";
      el.textContent = `Δ≤${fmt(th)}`;
      badge.appendChild(el);
    }
  }

  function loadScenario() {
    const sel = document.getElementById("scenario").value;
    const ex = preset(sel);
    document.getElementById("baseline").value = ex.baseline;
    document.getElementById("candidate").value = ex.candidate;
  }

  // Wire events
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("loadScenario").addEventListener("click", loadScenario);
    document.getElementById("run").addEventListener("click", run);
    document.getElementById("reset").addEventListener("click", () => {
      document.getElementById("candidate").value = "";
      document.getElementById("hBase").textContent = "–";
      document.getElementById("hX").textContent = "–";
      document.getElementById("delta").textContent = "–";
      document.getElementById("status").textContent = "–";
      document.getElementById("badge").innerHTML = "";
      document.getElementById("ncd").textContent = "–";
    });
    document.getElementById("threshold").addEventListener("input", (e) => {
      document.getElementById("thVal").textContent = parseFloat(e.target.value).toFixed(2);
    });
    // Load default example and compute once
    loadScenario();
    run();
  });
})();

