// Greedy auto-evict selection used by the weight test page. Runs in the
// browser and under node, so the page and the command-line check share one
// implementation.
//
// New surface is read from per-point coverage bitmasks precomputed by the data
// script: bit j of a point is set when that point lies within the radius of
// candidate j's cloud. A point is "new" against a selected set S exactly when
// its bitmask shares no bit with S, so no point cloud has to reach the page.
(function (root) {
  const ANGLE_FNS = {
    cos: { label: '(1 − cos θ) / 2', f: (t) => (1 - Math.cos(t * Math.PI / 180)) / 2 },
    linear: { label: 'θ / 180°', f: (t) => t / 180 },
    sat: { label: 'min(θ / θ₀, 1)', f: (t, t0) => Math.min(t / t0, 1) },
  };

  function newSurface(hist, selMask) {
    let total = 0, fresh = 0;
    for (const [bits, count] of hist) {
      total += count;
      if ((bits & selMask) === 0) fresh += count;
    }
    return fresh / total;
  }

  function select(data, cfg) {
    const cands = data.candidates;
    const hist = data.hist[String(cfg.radius)];
    const sum = cfg.wa + cfg.wv + cfg.wc;
    const w = sum > 0 ? { a: cfg.wa / sum, v: cfg.wv / sum, c: cfg.wc / sum } : { a: 1, v: 0, c: 0 };
    const angle = ANGLE_FNS[cfg.fn].f;
    const maxArea = Math.max(...cands.map((c) => c.area));
    const pool = cands.map((_, i) => i);
    const sel = [];
    let selMask = 0;
    const rounds = [];
    while (sel.length < data.limit && pool.length) {
      const rows = pool.map((i) => {
        const A = cands[i].area / maxArea;
        // Round 1 has nothing selected: every point is new surface (N = 1) and
        // there is no view to compare against (V is undefined, scored 0). Both
        // are the same for every candidate, so area alone ranks round 1.
        let theta = null, V = null, N = 1;
        if (sel.length) {
          theta = Math.min(...sel.map((j) => data.theta[i][j]));
          V = angle(theta, cfg.theta0);
          N = newSurface(hist[i], selMask);
        }
        return { i, A, theta, V, N, total: w.a * A + w.v * (V || 0) + w.c * N };
      });
      rows.sort((x, y) => y.total - x.total || y.A - x.A);
      const pick = rows[0];
      sel.push(pick.i);
      selMask |= 1 << pick.i;
      pool.splice(pool.indexOf(pick.i), 1);
      rounds.push({ pick, rows });
    }
    // Share of all candidate points within the radius of the kept set; a kept
    // candidate covers itself entirely.
    let covered = 0, all = 0;
    cands.forEach((c, i) => {
      all += c.points;
      covered += sel.includes(i) ? c.points : c.points * (1 - newSurface(hist[i], selMask));
    });
    const parts = {};
    sel.forEach((i) => { parts[cands[i].part] = (parts[cands[i].part] || 0) + 1; });
    return { sel, rounds, weights: w, coverage: covered / all, parts };
  }

  const api = { ANGLE_FNS, select, newSurface };
  if (typeof module !== 'undefined') module.exports = api;
  else root.EvictAlgo = api;
})(this);
