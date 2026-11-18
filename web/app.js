// BleedAlert in the browser: same model, preprocessing, triage rules and heatmap as the
// Python app, run locally with ONNX Runtime Web. Nothing is uploaded.
const LABELS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'];
const URGENT = 0.65, UNSURE_LOW = 0.35; // same thresholds as app.py
const ORDER = { URGENT: 0, 'NEEDS HUMAN REVIEW': 1, ROUTINE: 2 };
const SAMPLES = ['synthetic_intraparenchymal.dcm', 'synthetic_intraventricular.dcm', 'synthetic_normal.dcm', 'synthetic_subdural.dcm'];

ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1; // GitHub Pages has no cross-origin isolation
let session = null;
const studies = new Map(); // study key -> case
let selected = null;
const alerted = new Set();

const $ = (id) => document.getElementById(id);
const triage = (p) => (p >= URGENT ? 'URGENT' : p >= UNSURE_LOW ? 'NEEDS HUMAN REVIEW' : 'ROUTINE');
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.005 ? 0 : 1)}%`;

function summarize(p, subtype, status) {
  // Same template the Python app uses when no Gemini key is set.
  return `Triage ${status}. Model estimates ${Math.round(p * 100)}% probability of intracranial hemorrhage` +
    (status !== 'ROUTINE' ? `, most consistent with ${subtype}.` : '.') +
    ' This is triage support only. A physician must review the images.';
}

async function init() {
  try {
    session = await ort.InferenceSession.create('model.onnx', { executionProviders: ['wasm'] });
    $('mode').textContent = 'Model loaded · runs in your browser';
    $('loadDemo').disabled = false;
  } catch (e) {
    $('mode').textContent = 'Could not load the model: ' + e.message;
  }
}

async function analyze(buffer, name) {
  const pre = BleedPre.fromDicom(buffer, dicomParser);
  const out = await session.run({ image: new ort.Tensor('float32', pre.tensor, [1, 3, 256, 256]) });
  const probs = Array.from(out.probs.data);
  const scores = Object.fromEntries(LABELS.map((l, i) => [l, probs[i]]));
  const subtype = LABELS.slice(1).reduce((a, b) => (scores[b] > scores[a] ? b : a));
  const status = triage(scores.any);
  const key = pre.studyUid || name;
  const slice = { name, scores, subtype, status, rgb: pre.rgb, cam: Float32Array.from(out.cam.data) };
  const existing = studies.get(key);
  if (!existing) {
    studies.set(key, { key, patient: pre.patient || name, added: Date.now(), slices: 1, best: slice, reviewed: false });
  } else {
    // A study's score is its most suspicious slice, like the server version.
    existing.slices += 1;
    if (scores.any > existing.best.scores.any) existing.best = slice;
  }
}

async function addFiles(files) {
  const errors = [];
  for (const f of files) {
    try {
      await analyze(await f.arrayBuffer(), f.name);
    } catch (e) {
      errors.push(`${f.name}: ${e.message}`);
    }
    render();
  }
  if (errors.length) $('drop').innerHTML = `Some files could not be read:<br><small>${errors.map(escapeHtml).join('<br>')}</small>`;
}

function queue() {
  return [...studies.values()].filter((s) => !s.reviewed)
    .sort((a, b) => ORDER[a.best.status] - ORDER[b.best.status] || a.added - b.added);
}

function render() {
  const items = $('items');
  items.innerHTML = '';
  for (const s of queue()) {
    const b = s.best;
    const el = document.createElement('div');
    el.className = `item ${b.status.split(' ')[0]} ${selected === s.key ? 'active' : ''}`;
    el.tabIndex = 0;
    el.innerHTML = `<span class="dot"></span><div><b>${b.status} · ${pct(b.scores.any)}</b>
      <small>${s.slices} slice${s.slices > 1 ? 's' : ''} · ${escapeHtml(b.name)}</small></div>`;
    el.onclick = () => show(s.key);
    el.onkeydown = (e) => { if (e.key === 'Enter') show(s.key); };
    items.appendChild(el);
    if (b.status === 'URGENT' && !alerted.has(s.key)) {
      alerted.add(s.key);
      speak(`Urgent head CT. Likely ${b.subtype} bleed. Please review now.`);
    }
  }
  if (!selected && queue().length) show(queue()[0].key);
}

function draw(canvas, rgb, cam) {
  const size = 256;
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    let r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    if (cam) {
      // Bilinear upsample of the 8x8 map, then a red overlay at 55% max opacity (as model.overlay()).
      const y = Math.floor(i / size), x = i % size;
      const a = 0.55 * bilinear(cam, 8, (x + 0.5) * 8 / size - 0.5, (y + 0.5) * 8 / size - 0.5);
      r = r * (1 - a) + 255 * a; g *= 1 - a; b *= 1 - a;
    }
    img.data.set([r, g, b, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

function bilinear(m, n, x, y) {
  x = Math.min(Math.max(x, 0), n - 1); y = Math.min(Math.max(y, 0), n - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, n - 1), y1 = Math.min(y0 + 1, n - 1);
  const fx = x - x0, fy = y - y0;
  return (m[y0 * n + x0] * (1 - fx) + m[y0 * n + x1] * fx) * (1 - fy) + (m[y1 * n + x0] * (1 - fx) + m[y1 * n + x1] * fx) * fy;
}

function show(key) {
  selected = key;
  const s = studies.get(key);
  const b = s.best;
  const rows = LABELS.map((l) => `<span>${l === 'any' ? '<b>Any bleed</b>' : l}</span>
    <span class="bar"><i style="width:${(b.scores[l] * 100).toFixed(1)}%;${b.scores[l] >= URGENT ? 'background:var(--urgent)' : ''}"></i></span>
    <span>${pct(b.scores[l])}</span>`).join('');
  $('detail').innerHTML = `
    <span class="status s-${b.status.split(' ')[0]}">${b.status}</span>
    <h2>${escapeHtml(b.name)}</h2>
    <p class="note">Most suspicious of ${s.slices} slice${s.slices > 1 ? 's' : ''} in this study</p>
    <div class="imgs">
      <figure><canvas id="cScan" aria-label="CT slice"></canvas><figcaption>Scan</figcaption></figure>
      <figure><canvas id="cHeat" aria-label="CT slice with heatmap"></canvas><figcaption>Where the model is looking</figcaption></figure>
    </div>
    <p>Bleed probability <b>${pct(b.scores.any)}</b> · most likely subtype <b>${b.subtype}</b></p>
    <div class="summary">${summarize(b.scores.any, b.subtype, b.status)}</div>
    <h3 style="font-size:14px;color:var(--quiet);margin:20px 0 8px">All scores</h3>
    <div class="scores">${rows}</div>
    <p style="margin-top:20px"><button id="rev">Mark reviewed</button>
      <button id="say" style="background:var(--panel);color:var(--ink);border:1px solid var(--line)">Read aloud</button></p>
    <p class="note">Triage support only. Not a diagnosis. A physician reviews every scan.
      The full app adds Gemini-written summaries and a SQLite worklist.</p>`;
  draw($('cScan'), b.rgb, null);
  draw($('cHeat'), b.rgb, b.cam);
  $('rev').onclick = () => { s.reviewed = true; selected = null; $('detail').innerHTML = '<p class="note">Reviewed. Pick the next case.</p>'; render(); };
  $('say').onclick = () => speak(summarize(b.scores.any, b.subtype, b.status));
  render();
}

function speak(text) {
  if (!('speechSynthesis' in window)) return;
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function escapeHtml(t) {
  return String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── Queueing simulation, ported from simulate.py (same algorithm and defaults) ──
function rng(seed) { // mulberry32
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function simulate(perDay = 120, bleedRate = 0.1, readMin = 8, sens = 0.9, spec = 0.85, days = 200) {
  const r = rng(0), waits = { fifo: [], triage: [] };
  for (let d = 0; d < days; d++) {
    const scans = Array.from({ length: perDay }, () => r() * 1440).sort((a, b) => a - b).map((t) => {
      const bleed = r() < bleedRate; return [t, bleed, r() < (bleed ? sens : 1 - spec)];
    });
    for (const mode of ['fifo', 'triage']) {
      let clock = 0, i = 0; const q = [];
      while (i < scans.length || q.length) {
        if (!q.length && i < scans.length && scans[i][0] > clock) clock = scans[i][0];
        while (i < scans.length && scans[i][0] <= clock) {
          const [t, bleed, flagged] = scans[i++]; q.push([mode === 'triage' && flagged ? 0 : 1, t, bleed]);
        }
        q.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const [, t, bleed] = q.shift();
        if (bleed) waits[mode].push(clock - t);
        clock += -Math.log(1 - r()) * readMin;
      }
    }
  }
  const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  return { fifo: avg(waits.fifo), triage: avg(waits.triage) };
}
function sim() {
  const n = +$('load').value; $('loadv').textContent = n;
  const r = simulate(n);
  $('simout').innerHTML = `Average wait for a bleed patient: <b>${r.fifo.toFixed(1)} min</b> first come first served vs ` +
    `<b>${r.triage.toFixed(1)} min</b> with BleedAlert. <b>${(r.fifo - r.triage).toFixed(1)} minutes sooner.</b>`;
}

// ── Wiring ──
const drop = $('drop'), input = $('file');
drop.onclick = () => input.click();
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); };
input.onchange = (e) => addFiles(e.target.files);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); };
$('loadDemo').onclick = async (e) => {
  e.target.disabled = true; e.target.textContent = 'Scoring…';
  for (const name of SAMPLES) {
    const res = await fetch('demo/' + name);
    await analyze(await res.arrayBuffer(), name);
  }
  render();
  e.target.disabled = false; e.target.textContent = 'Load 4 sample scans';
};
$('load').oninput = sim;
setTimeout(sim, 0);
init();
