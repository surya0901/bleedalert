// BleedAlert preprocessing for the browser. Mirrors ct_utils.py exactly:
//   DICOM -> Hounsfield units -> 3 CT windows -> uint8 RGB -> Pillow-style bicubic
//   resize to 256x256 -> ImageNet normalization -> NCHW float32.
// Works in browsers (window.BleedPre) and in Node (module.exports) for parity tests.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.BleedPre = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  const SIZE = 256;
  const WINDOWS = [[40, 80], [80, 200], [600, 2800]]; // brain, subdural, bone
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
  const UNCOMPRESSED = new Set(['1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2']);

  function num(ds, tag, fallback) {
    const s = ds.string(tag);
    if (s === undefined || s === '') return fallback;
    const v = parseFloat(s.split('\\')[0]);
    return Number.isFinite(v) ? v : fallback;
  }

  /** Parse a DICOM ArrayBuffer into {hu, rows, cols, studyUid, patient}. */
  function parseDicom(buffer, dicomParser) {
    const bytes = new Uint8Array(buffer);
    const ds = dicomParser.parseDicom(bytes);
    const ts = ds.string('x00020010');
    if (ts && !UNCOMPRESSED.has(ts.trim())) {
      throw new Error('Compressed DICOM is not supported in the browser demo. Export the series uncompressed.');
    }
    const rows = ds.uint16('x00280010');
    const cols = ds.uint16('x00280011');
    const bits = ds.uint16('x00280100');
    const signed = ds.uint16('x00280103') === 1;
    const el = ds.elements.x7fe00010;
    if (!el || !rows || !cols) throw new Error('No pixel data found.');
    const n = rows * cols;
    let px;
    const off = el.dataOffset + bytes.byteOffset;
    if (bits === 16) {
      const copy = bytes.buffer.slice(off, off + n * 2); // copy for alignment
      px = signed ? new Int16Array(copy) : new Uint16Array(copy);
    } else if (bits === 8) {
      px = bytes.subarray(el.dataOffset, el.dataOffset + n);
    } else {
      throw new Error(`Unsupported BitsAllocated=${bits}.`);
    }
    const slope = num(ds, 'x00281053', 1);
    const intercept = num(ds, 'x00281052', 0);
    const hu = new Float32Array(n);
    for (let i = 0; i < n; i++) hu[i] = px[i] * slope + intercept;
    return {
      hu, rows, cols,
      studyUid: ds.string('x0020000d') || null,
      patient: ds.string('x00100020') || null,
    };
  }

  /** HU -> interleaved uint8 RGB (one CT window per channel), like dicom_to_rgb(). */
  function windowRGB(hu, n) {
    const rgb = new Uint8ClampedArray(n * 3);
    for (let c = 0; c < 3; c++) {
      const [center, width] = WINDOWS[c];
      const lo = center - width / 2;
      const range = width;
      for (let i = 0; i < n; i++) {
        let v = (hu[i] - lo) / range;
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        rgb[i * 3 + c] = Math.floor(Math.fround(v) * 255); // numpy astype(uint8) truncates
      }
    }
    return rgb;
  }

  // Pillow's bicubic kernel (a = -0.5) and its separable, antialiased resample.
  function bicubic(x) {
    const a = -0.5;
    x = Math.abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  }

  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const fscale = Math.max(scale, 1);
    const support = 2 * fscale;
    const out = [];
    for (let i = 0; i < outSize; i++) {
      const center = (i + 0.5) * scale;
      const ss = 1 / fscale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
      const w = [];
      let total = 0;
      for (let x = xmin; x < xmax; x++) {
        const v = bicubic((x - center + 0.5) * ss);
        w.push(v);
        total += v;
      }
      out.push({ xmin, w: w.map((v) => (total ? v / total : 0)) });
    }
    return out;
  }

  const clip8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

  /** Resize interleaved RGB uint8 (h x w) to (oh x ow), horizontal pass then vertical, like Pillow. */
  function resizeRGB(src, h, w, oh, ow) {
    if (h === oh && w === ow) return src;
    const cx = coeffs(w, ow);
    const tmp = new Uint8ClampedArray(h * ow * 3);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < ow; x++) {
        const { xmin, w: k } = cx[x];
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let j = 0; j < k.length; j++) s += src[(y * w + xmin + j) * 3 + c] * k[j];
          tmp[(y * ow + x) * 3 + c] = clip8(s);
        }
      }
    }
    const cy = coeffs(h, oh);
    const out = new Uint8ClampedArray(oh * ow * 3);
    for (let y = 0; y < oh; y++) {
      const { xmin: ymin, w: k } = cy[y];
      for (let x = 0; x < ow; x++) {
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let j = 0; j < k.length; j++) s += tmp[((ymin + j) * ow + x) * 3 + c] * k[j];
          out[(y * ow + x) * 3 + c] = clip8(s);
        }
      }
    }
    return out;
  }

  /** uint8 RGB (SIZE x SIZE) -> normalized NCHW Float32Array. */
  function toTensor(rgb) {
    const n = SIZE * SIZE;
    const t = new Float32Array(3 * n);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < n; i++) t[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
    }
    return t;
  }

  /** Full pipeline: DICOM ArrayBuffer -> {tensor, rgb (256x256 RGB), studyUid, patient}. */
  function fromDicom(buffer, dicomParser) {
    const d = parseDicom(buffer, dicomParser);
    const rgb = resizeRGB(windowRGB(d.hu, d.rows * d.cols), d.rows, d.cols, SIZE, SIZE);
    return { tensor: toTensor(rgb), rgb, studyUid: d.studyUid, patient: d.patient };
  }

  return { SIZE, fromDicom, parseDicom, windowRGB, resizeRGB, toTensor };
});
