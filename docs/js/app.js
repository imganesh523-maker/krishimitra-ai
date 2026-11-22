/* KrishiMitra AI · in-browser inference with ONNX Runtime Web */
(() => {
  'use strict';

  const ORT_VER = '1.22.0';
  const MODELS = {
    int8: { url: 'model/krishimitra_int8.onnx', label: 'INT8', size: '3.0 MB' },
    fp32: { url: 'model/krishimitra.onnx', label: 'FP32', size: '9.3 MB' },
  };
  const SAMPLES = [
    '00_Tomato_Late_blight.jpg', '01_Potato_Late_blight.jpg', '02_Tomato_Yellow_Leaf_Curl_Virus.jpg',
    '03_Corn_Common_rust.jpg', '04_Apple_scab.jpg', '05_Squash_Powdery_mildew.jpg', '06_Grape_Esca.jpg',
    '07_Tomato_healthy.jpg', '08_Potato_Early_blight.jpg', '09_Tomato_Bacterial_spot.jpg',
  ];
  const SIZE = 160, RESIZE = 176;
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const LOW_CONF = 0.5;
  const RING_LEN = 2 * Math.PI * 52;

  const I18N = {
    en: {
      diag: 'Diagnosis', top3: 'Top-3 predictions', conf: 'confidence',
      cause: 'Cause', symptoms: 'Symptoms', treatment: 'Treatment', prevention: 'Prevention',
      fine: 'AI-based aid, not a final diagnosis. Confirm with your local Krishi Vigyan Kendra (KVK) before applying any chemical.',
      low: (p) => `Low confidence (${p}%). The model is unsure: retake the photo of a single leaf in good light on a plain background, and check the alternatives below.`,
      types: { healthy: 'healthy', fungal: 'fungal', 'fungal (oomycete)': 'fungal (oomycete)', bacterial: 'bacterial', viral: 'viral', pest: 'pest' },
    },
    hi: {
      diag: 'निदान', top3: 'शीर्ष 3 अनुमान', conf: 'विश्वास',
      cause: 'कारण', symptoms: 'लक्षण', treatment: 'उपचार', prevention: 'बचाव',
      fine: 'यह AI आधारित सहायता है, अंतिम निदान नहीं। कोई भी दवा डालने से पहले नज़दीकी कृषि विज्ञान केंद्र (KVK) से पुष्टि करें।',
      low: (p) => `कम विश्वास (${p}%)। मॉडल निश्चित नहीं है: एक पत्ती की साफ़ रोशनी और सादे पृष्ठभूमि में दोबारा फ़ोटो लें, और नीचे दिए विकल्प भी देखें।`,
      types: { healthy: 'स्वस्थ', fungal: 'फफूंद', 'fungal (oomycete)': 'फफूंद (ऊमाइसीट)', bacterial: 'जीवाणु', viral: 'विषाणु', pest: 'कीट' },
    },
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    nav: $('nav'), demo: $('demoPanel'), mstatus: $('mstatus'), mtext: $('mtext'), mprogress: $('mprogress'),
    viewer: $('viewer'), file: $('file'), photo: $('photo'), heat: $('heat'), scan: $('scan'), replace: $('replaceBtn'),
    heatCtl: $('heatCtl'), heatToggle: $('heatToggle'), opacity: $('opacity'), opVal: $('opVal'),
    samples: $('samples'), resultEmpty: $('resultEmpty'), result: $('result'),
    ringBar: $('ringBar'), confVal: $('confVal'), confLbl: $('confLbl'),
    kDiag: $('kDiag'), rPlant: $('rPlant'), rDisease: $('rDisease'), rType: $('rType'), rTime: $('rTime'),
    alert: $('alert'), kTop3: $('kTop3'), top3: $('top3'), tabs: $('tabs'), tabBody: $('tabBody'), fine: $('fine'),
  };

  const state = {
    prec: 'int8', sessions: {}, loading: {}, classes: null, kb: null,
    lang: 'en', tab: 'cause', last: null, current: null, runId: 0, sampleBtns: [],
  };

  /* ---------------- ORT setup ---------------- */
  function initOrt() {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web failed to load');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.logLevel = 'error';
  }

  function setStatus(kind, text) {
    el.mstatus.classList.remove('ready', 'error', 'loading');
    el.mstatus.classList.add(kind);
    el.mtext.textContent = text;
  }
  function setProgress(frac) {
    el.mprogress.classList.toggle('done', frac >= 1);
    el.mprogress.firstElementChild.style.width = `${Math.round(Math.min(1, frac) * 100)}%`;
  }

  async function fetchWithProgress(url, onProg) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const total = +res.headers.get('content-length') || 0;
    if (!res.body || !total) { const b = await res.arrayBuffer(); onProg(1); return b; }
    const reader = res.body.getReader();
    const parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length; onProg(Math.min(1, got / total));
    }
    const out = new Uint8Array(got); let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    onProg(1);
    return out.buffer;
  }

  function loadModel(prec) {
    if (state.sessions[prec]) return Promise.resolve(state.sessions[prec]);
    if (state.loading[prec]) return state.loading[prec];
    const m = MODELS[prec];
    setStatus('loading', `Downloading ${m.label} model · ${m.size}`);
    setProgress(0.02);
    const p = (async () => {
      const t0 = performance.now();
      const buf = await fetchWithProgress(m.url, (f) => {
        setProgress(0.02 + f * 0.88);
        el.mtext.textContent = `Downloading ${m.label} model · ${Math.round(f * 100)}%`;
      });
      el.mtext.textContent = `Initialising ${m.label} runtime…`;
      setProgress(0.95);
      const sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      state.sessions[prec] = sess;
      setProgress(1);
      if (state.prec === prec) setStatus('ready', `${m.label} model ready · ${m.size} · loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      return sess;
    })();
    state.loading[prec] = p;
    p.catch((e) => {
      delete state.loading[prec];
      setStatus('error', `Could not load model: ${e.message}`);
      console.error(e);
    });
    return p;
  }

  /* ---------------- image helpers ---------------- */
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not decode image'));
      img.src = src;
    });
  }

  // Resize short side to 176 then centre-crop 160 (same as torchvision eval transform).
  function preprocess(img) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const s = RESIZE / Math.min(w, h);
    const rw = Math.max(SIZE, Math.round(w * s)), rh = Math.max(SIZE, Math.round(h * s));
    const left = Math.floor((rw - SIZE) / 2), top = Math.floor((rh - SIZE) / 2);
    const c = document.createElement('canvas');
    c.width = rw; c.height = rh;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, rw, rh);
    const px = ctx.getImageData(left, top, SIZE, SIZE).data;
    const n = SIZE * SIZE, data = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      data[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0];
      data[n + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
      data[2 * n + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    // the same crop region in source pixels, for the high-res display canvas
    const crop = { x: left / s, y: top / s, w: SIZE / s, h: SIZE / s };
    return { tensor: new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]), crop };
  }

  function drawPhoto(img, crop) {
    const ctx = el.photo.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, el.photo.width, el.photo.height);
    ctx.drawImage(img, crop.x, crop.y, crop.w, crop.h, 0, 0, el.photo.width, el.photo.height);
  }

  function softmax(arr) {
    let m = -Infinity; for (const v of arr) if (v > m) m = v;
    const e = Array.from(arr, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function jet(t) {
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
    return [r * 255, g * 255, b * 255];
  }

  // Normalise the class's 5x5 CAM, upsample bilinearly to 160x160, colour with jet.
  function drawCam(cams, dims, cls) {
    const H = dims[2], W = dims[3], off = cls * H * W;
    const m = new Float32Array(H * W);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < H * W; i++) { const v = Math.max(0, cams[off + i]); m[i] = v; if (v < lo) lo = v; if (v > hi) hi = v; }
    const rng = hi - lo || 1;
    for (let i = 0; i < m.length; i++) m[i] = (m[i] - lo) / rng;
    const ctx = el.heat.getContext('2d');
    const out = ctx.createImageData(SIZE, SIZE);
    for (let y = 0; y < SIZE; y++) {
      const sy = Math.min(H - 1, Math.max(0, (y + 0.5) * H / SIZE - 0.5));
      const y0 = Math.floor(sy), y1 = Math.min(H - 1, y0 + 1), fy = sy - y0;
      for (let x = 0; x < SIZE; x++) {
        const sx = Math.min(W - 1, Math.max(0, (x + 0.5) * W / SIZE - 0.5));
        const x0 = Math.floor(sx), x1 = Math.min(W - 1, x0 + 1), fx = sx - x0;
        const v = (m[y0 * W + x0] * (1 - fx) + m[y0 * W + x1] * fx) * (1 - fy) +
                  (m[y1 * W + x0] * (1 - fx) + m[y1 * W + x1] * fx) * fy;
        const [r, g, b] = jet(v), o = (y * SIZE + x) * 4;
        out.data[o] = r; out.data[o + 1] = g; out.data[o + 2] = b; out.data[o + 3] = 255;
      }
    }
    ctx.putImageData(out, 0, 0);
  }

  async function infer(img) {
    const sess = await loadModel(state.prec);
    const { tensor, crop } = preprocess(img);
    const t0 = performance.now();
    const out = await sess.run({ input: tensor });
    const ms = performance.now() - t0;
    const probs = softmax(out.logits.data);
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    return { probs, order, cams: out.cams.data, camDims: out.cams.dims, crop, ms };
  }

  /* ---------------- UI flow ---------------- */
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function analyse(src, sampleIdx = -1) {
    const id = ++state.runId;
    state.sampleBtns.forEach((b, i) => b.classList.toggle('active', i === sampleIdx));
    let img;
    try { img = await loadImage(src); } catch (e) { flashError(e.message); return; }
    if (id !== state.runId) return;
    state.current = { img, src, sampleIdx };
    el.viewer.classList.add('has-img');
    el.heat.style.opacity = 0;
    const { crop } = cropOnly(img);
    drawPhoto(img, crop);
    el.scan.classList.add('on');
    el.viewer.setAttribute('aria-busy', 'true');
    try {
      const [res] = await Promise.all([infer(img), sleep(750), ready]);
      if (id !== state.runId) return;
      drawCam(res.cams, res.camDims, res.order[0][1]);
      state.last = res;
      render();
      applyHeat();
    } catch (e) {
      if (id === state.runId) flashError(e.message);
      console.error(e);
    } finally {
      if (id === state.runId) { el.scan.classList.remove('on'); el.viewer.removeAttribute('aria-busy'); }
    }
  }

  function cropOnly(img) {
    const w = img.naturalWidth, h = img.naturalHeight, s = RESIZE / Math.min(w, h);
    const rw = Math.max(SIZE, Math.round(w * s)), rh = Math.max(SIZE, Math.round(h * s));
    const left = Math.floor((rw - SIZE) / 2), top = Math.floor((rh - SIZE) / 2);
    return { crop: { x: left / s, y: top / s, w: SIZE / s, h: SIZE / s } };
  }

  function flashError(msg) {
    el.resultEmpty.hidden = false; el.result.hidden = true;
    el.resultEmpty.querySelector('h3').textContent = 'Something went wrong';
    el.resultEmpty.querySelector('p').textContent = msg;
  }

  function info(cls) {
    return state.kb[cls] || { plant_en: cls.split('___')[0], disease_en: (cls.split('___')[1] || '').replace(/_/g, ' '), type: '' };
  }
  const L = (obj, key) => obj[`${key}_${state.lang}`] ?? obj[`${key}_en`] ?? '';

  function render() {
    const res = state.last; if (!res) return;
    const t = I18N[state.lang];
    const [p1, c1] = res.order[0];
    const cls = state.classes[c1], k = info(cls);
    const pct = Math.round(p1 * 1000) / 10;

    el.resultEmpty.hidden = true;
    el.result.hidden = false;
    el.result.classList.toggle('hi-mode', state.lang === 'hi');
    el.result.lang = state.lang;

    el.kDiag.textContent = t.diag; el.kTop3.textContent = t.top3; el.confLbl.textContent = t.conf;
    ['cause', 'symptoms', 'treatment', 'prevention'].forEach((x) => { $(`t${x[0].toUpperCase()}${x.slice(1)}`).textContent = t[x]; });
    el.fine.textContent = t.fine;

    el.rPlant.textContent = L(k, 'plant');
    el.rDisease.textContent = L(k, 'disease');
    el.rType.className = `badge t-${(k.type || '').split(' ')[0]}`;
    el.rType.textContent = t.types[k.type] || k.type || '—';
    el.rTime.textContent = `${res.ms.toFixed(0)} ms · ${MODELS[state.prec].label}`;

    const color = p1 >= 0.8 ? 'var(--em)' : p1 >= LOW_CONF ? 'var(--lime)' : 'var(--amber)';
    el.ringBar.style.stroke = color; el.ringBar.style.color = color;
    el.ringBar.style.strokeDashoffset = RING_LEN;
    requestAnimationFrame(() => requestAnimationFrame(() => { el.ringBar.style.strokeDashoffset = RING_LEN * (1 - p1); }));
    countTo(el.confVal, pct, 1);

    if (p1 < LOW_CONF) { el.alert.hidden = false; el.alert.textContent = t.low(pct.toFixed(1)); }
    else el.alert.hidden = true;

    el.top3.innerHTML = '';
    res.order.slice(0, 3).forEach(([p, i]) => {
      const kk = info(state.classes[i]);
      const row = document.createElement('div'); row.className = 't3';
      row.innerHTML = `<div class="t3-row"><span class="t3-name"></span><span class="t3-pct">${(p * 100).toFixed(1)}%</span></div><div class="t3-bar"><i></i></div>`;
      const nm = row.querySelector('.t3-name');
      nm.textContent = L(kk, 'disease');
      const sm = document.createElement('small'); sm.textContent = L(kk, 'plant'); nm.appendChild(sm);
      el.top3.appendChild(row);
      requestAnimationFrame(() => requestAnimationFrame(() => { row.querySelector('i').style.width = `${Math.max(1.5, p * 100)}%`; }));
    });
    renderTab();
  }

  function renderTab() {
    const res = state.last; if (!res) return;
    const k = info(state.classes[res.order[0][1]]);
    const v = L(k, state.tab);
    el.tabBody.innerHTML = '';
    if (Array.isArray(v)) {
      const ul = document.createElement('ul');
      v.forEach((s) => { const li = document.createElement('li'); li.textContent = s; ul.appendChild(li); });
      el.tabBody.appendChild(ul);
    } else {
      const p = document.createElement('p'); p.textContent = v || '—'; el.tabBody.appendChild(p);
    }
    el.tabBody.style.animation = 'none'; void el.tabBody.offsetWidth; el.tabBody.style.animation = '';
  }

  function countTo(node, target, dec = 0, dur = 900) {
    const t0 = performance.now(), from = 0;
    const step = (now) => {
      const f = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - f, 3);
      node.textContent = (from + (target - from) * e).toFixed(dec);
      if (f < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function applyHeat() {
    const v = +el.opacity.value;
    el.opVal.textContent = `${v}%`;
    el.opacity.style.setProperty('--p', `${v}%`);
    el.heat.style.opacity = state.last && el.heatToggle.checked ? v / 100 : 0;
  }

  /* ---------------- wiring ---------------- */
  function buildSamples() {
    SAMPLES.forEach((f, i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'sample';
      const name = f.replace(/^\d+_/, '').replace(/\.\w+$/, '').replace(/_/g, ' ');
      b.title = name; b.setAttribute('aria-label', `Sample: ${name}`);
      b.innerHTML = `<img src="samples/${f}" alt="" loading="lazy"><span></span>`;
      b.querySelector('span').textContent = name;
      b.addEventListener('click', () => analyse(`samples/${f}`, i));
      el.samples.appendChild(b);
      state.sampleBtns.push(b);
    });
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    analyse(url, -1);
  }

  function wire() {
    el.viewer.addEventListener('click', (e) => {
      if (e.target === el.replace || !el.viewer.classList.contains('has-img')) el.file.click();
    });
    el.replace.addEventListener('click', (e) => { e.stopPropagation(); el.file.click(); });
    el.viewer.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); } });
    el.file.addEventListener('change', () => { handleFile(el.file.files[0]); el.file.value = ''; });
    ['dragenter', 'dragover'].forEach((ev) => el.viewer.addEventListener(ev, (e) => { e.preventDefault(); el.viewer.classList.add('drag'); }));
    ['dragleave', 'dragend'].forEach((ev) => el.viewer.addEventListener(ev, () => el.viewer.classList.remove('drag')));
    el.viewer.addEventListener('drop', (e) => {
      e.preventDefault(); el.viewer.classList.remove('drag');
      handleFile(e.dataTransfer.files && e.dataTransfer.files[0]);
    });
    window.addEventListener('paste', (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
      if (item) handleFile(item.getAsFile());
    });

    document.querySelectorAll('[data-prec]').forEach((b) => b.addEventListener('click', async () => {
      const prec = b.dataset.prec;
      if (prec === state.prec) return;
      document.querySelectorAll('[data-prec]').forEach((x) => x.classList.toggle('active', x === b));
      state.prec = prec;
      try {
        await loadModel(prec);
        const m = MODELS[prec];
        if (state.prec === prec) setStatus('ready', `${m.label} model ready · ${m.size}`);
        if (state.current && state.prec === prec) analyse(state.current.src, state.current.sampleIdx);
      } catch (_) { /* status already shows the error */ }
    }));

    document.querySelectorAll('[data-lang]').forEach((b) => b.addEventListener('click', () => {
      state.lang = b.dataset.lang;
      document.querySelectorAll('[data-lang]').forEach((x) => x.classList.toggle('active', x === b));
      render();
    }));

    el.tabs.addEventListener('click', (e) => {
      const b = e.target.closest('.tab'); if (!b) return;
      state.tab = b.dataset.tab;
      el.tabs.querySelectorAll('.tab').forEach((x) => { x.classList.toggle('active', x === b); x.setAttribute('aria-selected', x === b); });
      renderTab();
    });

    el.opacity.addEventListener('input', applyHeat);
    el.heatToggle.addEventListener('change', applyHeat);

    // nav shadow, spotlight, reveal, counters
    const onScroll = () => el.nav.classList.toggle('scrolled', window.scrollY > 10);
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
    el.demo.addEventListener('pointermove', (e) => {
      const r = el.demo.getBoundingClientRect();
      el.demo.style.setProperty('--mx', `${e.clientX - r.left}px`);
      el.demo.style.setProperty('--my', `${e.clientY - r.top}px`);
    });

    const counters = new WeakSet();
    const io = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        en.target.classList.add('in');
        en.target.querySelectorAll('[data-count]').forEach((n) => {
          if (counters.has(n)) return; counters.add(n);
          const target = +n.dataset.count, dec = +(n.dataset.dec || 0), sep = n.dataset.sep;
          const t0 = performance.now();
          const step = (now) => {
            const f = Math.min(1, (now - t0) / 1400), e = 1 - Math.pow(1 - f, 3), v = target * e;
            n.textContent = sep ? Math.round(v).toLocaleString('en-IN') : v.toFixed(dec);
            if (f < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        });
        io.unobserve(en.target);
      });
    }, { threshold: 0.12 }) : null;
    document.querySelectorAll('.reveal').forEach((n) => (io ? io.observe(n) : n.classList.add('in')));
  }

  /* ---------------- boot ---------------- */
  let readyResolve;
  const ready = new Promise((r) => { readyResolve = r; });
  window.__krishimitraReady = ready;

  // Test hook: returns the top-1 prediction for an image URL using the current precision.
  window.__krishimitraTest = async (url) => {
    await ready;
    const img = await loadImage(url);
    const res = await infer(img);
    const [p, i] = res.order[0];
    return { label: state.classes[i], prob: p };
  };

  async function boot() {
    buildSamples();
    wire();
    applyHeat();
    try {
      initOrt();
      const kbP = fetch('kb.json').then((r) => { if (!r.ok) throw new Error(`kb.json HTTP ${r.status}`); return r.json(); });
      const [kb] = await Promise.all([kbP, loadModel('int8')]);
      state.classes = kb.classes; state.kb = kb.kb;
      readyResolve(true);
    } catch (e) {
      setStatus('error', `Could not start: ${e.message}`);
      console.error(e);
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
