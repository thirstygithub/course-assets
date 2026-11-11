// OpenGL 1.x fixed-function emulation on WebGL2, for the subset the Facade
// engine uses: immediate mode, matrix stacks, 1D evaluators, textures,
// blending, alpha test, depth test. Vertices are transformed to clip space on
// the CPU so consecutive glBegin/glEnd blocks with the same raster state are
// drawn in one call.

const E = {
  MODELVIEW: 0x1700, PROJECTION: 0x1701, TEXTURE: 0x1702,
  TEXTURE_2D: 0x0de1, BLEND: 0x0be2, ALPHA_TEST: 0x0bc0, DEPTH_TEST: 0x0b71, CULL_FACE: 0x0b44,
  MAP1_VERTEX_3: 0x0d97, MAP1_VERTEX_4: 0x0d98, MAP1_COLOR_4: 0x0d90, MAP1_TEXTURE_COORD_2: 0x0d94,
  MODELVIEW_MATRIX: 0x0ba6, PROJECTION_MATRIX: 0x0ba7, VIEWPORT: 0x0ba2,
  CW: 0x0900, CCW: 0x0901,
};

const ident = () => new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

// A growable float buffer: `a` holds `n` used floats.
const floatBuf = () => ({ a: new Float32Array(1 << 14), n: 0 });
// Reserves k more floats in buf and returns their offset.
function grow(buf, k) {
  const o = buf.n;
  if (o + k > buf.a.length) {
    const a = new Float32Array(Math.max(buf.a.length * 2, o + k));
    a.set(buf.a.subarray(0, o));
    buf.a = a;
  }
  buf.n = o + k;
  return o;
}

function mul(a, b) {          // column-major a * b
  const r = new Float64Array(16);
  for (let c = 0; c < 4; c++)
    for (let row = 0; row < 4; row++)
      r[c * 4 + row] = a[row] * b[c * 4] + a[4 + row] * b[c * 4 + 1] + a[8 + row] * b[c * 4 + 2] + a[12 + row] * b[c * 4 + 3];
  return r;
}

const VS = `#version 300 es
in vec4 a_pos; in vec4 a_col; in vec2 a_uv;
out vec4 v_col; out vec2 v_uv;
uniform float u_pointSize;
void main() { gl_Position = a_pos; v_col = a_col; v_uv = a_uv; gl_PointSize = u_pointSize; }`;
const FS = `#version 300 es
precision highp float;
in vec4 v_col; in vec2 v_uv; out vec4 o;
uniform sampler2D u_tex; uniform bool u_useTex; uniform int u_alphaFunc; uniform float u_alphaRef;
void main() {
  vec4 c = v_col;
  if (u_useTex) c *= texture(u_tex, v_uv);
  float a = c.a, r = u_alphaRef;
  bool pass = u_alphaFunc == 0x207 || (u_alphaFunc == 0x201 && a < r) || (u_alphaFunc == 0x202 && a == r) ||
    (u_alphaFunc == 0x203 && a <= r) || (u_alphaFunc == 0x204 && a > r) || (u_alphaFunc == 0x205 && a != r) ||
    (u_alphaFunc == 0x206 && a >= r);
  if (!pass) discard;
  o = c;
}`;

const FLOATS_PER_VERTEX = 10;   // clip xyzw, rgba, st

export class GLEmu {
  constructor(canvas, memory) {
    this.canvas = canvas;
    this.memory = memory;
    // The engine draws into an offscreen (multisampled) framebuffer, copied to
    // the canvas only at SwapBuffers (present()): the engine can pause
    // mid-frame (Sleep, file loads, waits on the AI) and the browser would
    // otherwise show the half-drawn frame, as flashing.
    const gl = this.gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true, alpha: false, depth: false });
    if (!gl) throw new Error('WebGL2 not available');
    this.target = null;
    this.ensureTarget();
    this.prog = this.program(VS, FS);
    this.loc = {
      pos: gl.getAttribLocation(this.prog, 'a_pos'), col: gl.getAttribLocation(this.prog, 'a_col'),
      uv: gl.getAttribLocation(this.prog, 'a_uv'),
      useTex: gl.getUniformLocation(this.prog, 'u_useTex'), alphaFunc: gl.getUniformLocation(this.prog, 'u_alphaFunc'),
      alphaRef: gl.getUniformLocation(this.prog, 'u_alphaRef'), pointSize: gl.getUniformLocation(this.prog, 'u_pointSize'),
    };
    this.vbo = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    const stride = FLOATS_PER_VERTEX * 4;
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 4, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.loc.col);
    gl.vertexAttribPointer(this.loc.col, 4, gl.FLOAT, false, stride, 16);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.vertexAttribPointer(this.loc.uv, 2, gl.FLOAT, false, stride, 32);
    gl.useProgram(this.prog);
    gl.uniform1i(gl.getUniformLocation(this.prog, 'u_tex'), 0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    this.white = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.white);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));

    this.stacks = { [E.MODELVIEW]: [ident()], [E.PROJECTION]: [ident()], [E.TEXTURE]: [ident()] };
    this.mode = E.MODELVIEW;
    this.color = [1, 1, 1, 1];
    this.uv = [0, 0];
    this.enabled = new Set();
    this.blend = [gl.ONE, gl.ZERO];
    this.alpha = [0x207, 0];
    this.front = E.CCW;
    this.viewport = [0, 0, canvas.width, canvas.height];
    this.textures = new Map();
    this.nextTexture = 1;
    this.bound = 0;
    this.maps = new Map();     // target -> { u1, u2, order, comps, points }
    this.attribStack = [];
    this.lists = 1;
    this.listBase = 0;
    this.listGlyphs = new Map();   // display list id -> glyph (wglUseFontOutlines)
    this.prim = null;          // the glBegin mode while inside glBegin
    this.primVerts = floatBuf();
    this.batch = { tris: floatBuf(), lines: floatBuf(), points: floatBuf() };
    this.batchKey = '';
    this.stateDirty = true;
    this.mvpOf = [null, null];
    this.drawCalls = 0;
  }

  program(vs, fs) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  // (Re)creates the offscreen framebuffer at the canvas size and binds it.
  ensureTarget() {
    const { gl, canvas } = this;
    const w = canvas.width, h = canvas.height;
    if (this.target?.w === w && this.target?.h === h) return;
    if (this.target) {
      gl.deleteFramebuffer(this.target.fb);
      gl.deleteRenderbuffer(this.target.color);
      gl.deleteRenderbuffer(this.target.depth);
      gl.deleteFramebuffer(this.target.resolveFb);
      gl.deleteRenderbuffer(this.target.resolveColor);
    }
    // Resolving multisampling needs identical formats; the canvas (no alpha)
    // differs, so resolve into this single-sampled buffer and copy that.
    const resolveColor = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, resolveColor);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, w, h);
    const resolveFb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, resolveFb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, resolveColor);
    const samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES));
    const color = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, color);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, w, h);
    const depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH24_STENCIL8, w, h);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth);
    this.target = { fb, color, depth, resolveFb, resolveColor, w, h };
  }

  // SwapBuffers: draw what's pending, then copy the finished frame to the
  // canvas.
  present() {
    const { gl } = this;
    this.flush();
    this.ensureTarget();
    const { fb, resolveFb, w, h } = this.target;
    const scissor = gl.isEnabled(gl.SCISSOR_TEST);
    if (scissor) gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resolveFb);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, resolveFb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    if (scissor) gl.enable(gl.SCISSOR_TEST);
  }

  // A view of guest memory, remade only when the memory grows.
  dv() {
    if (this.dvBuf !== this.memory.buffer) { this.dvBuf = this.memory.buffer; this.dvCache = new DataView(this.dvBuf); }
    return this.dvCache;
  }

  get top() { const s = this.stacks[this.mode]; return s[s.length - 1]; }
  set top(m) { const s = this.stacks[this.mode]; s[s.length - 1] = m; }
  mv() { const s = this.stacks[E.MODELVIEW]; return s[s.length - 1]; }
  proj() { const s = this.stacks[E.PROJECTION]; return s[s.length - 1]; }
  texm() { const s = this.stacks[E.TEXTURE]; return s[s.length - 1]; }

  // ---- state that affects rasterization: changing it flushes the batch ----
  stateKey() {
    const tex = this.enabled.has(E.TEXTURE_2D) ? this.bound : 0;
    return [tex, this.enabled.has(E.BLEND) ? this.blend.join() : 'nb', this.enabled.has(E.ALPHA_TEST) ? this.alpha.join() : 'na',
      this.enabled.has(E.DEPTH_TEST), this.enabled.has(E.CULL_FACE) ? this.front : 'nc', this.viewport.join()].join('|');
  }

  // The raster state a batch is drawn with, captured when the batch starts
  // (later state changes belong to later batches).
  snapshot() {
    return {
      tex: this.enabled.has(E.TEXTURE_2D) && this.bound ? this.textures.get(this.bound) : null,
      blend: this.enabled.has(E.BLEND) ? [...this.blend] : null,
      alpha: this.enabled.has(E.ALPHA_TEST) ? [...this.alpha] : [0x207, 0],
      depth: this.enabled.has(E.DEPTH_TEST),
      cull: this.enabled.has(E.CULL_FACE) ? this.front : null,
      viewport: [...this.viewport],
    };
  }

  flush() {
    const { gl, batch } = this;
    if (!batch.tris.n && !batch.lines.n && !batch.points.n) return;
    const st = this.batchState;
    gl.viewport(...st.viewport);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, st.tex || this.white);
    gl.uniform1i(this.loc.useTex, st.tex ? 1 : 0);
    gl.uniform1i(this.loc.alphaFunc, st.alpha[0]);
    gl.uniform1f(this.loc.alphaRef, st.alpha[1]);
    gl.uniform1f(this.loc.pointSize, 1);
    if (st.blend) { gl.enable(gl.BLEND); gl.blendFunc(st.blend[0], st.blend[1]); } else gl.disable(gl.BLEND);
    if (st.depth) gl.enable(gl.DEPTH_TEST); else gl.disable(gl.DEPTH_TEST);
    if (st.cull !== null) { gl.enable(gl.CULL_FACE); gl.frontFace(st.cull === E.CW ? gl.CW : gl.CCW); } else gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    for (const [buf, mode] of [[batch.tris, gl.TRIANGLES], [batch.lines, gl.LINES], [batch.points, gl.POINTS]]) {
      if (!buf.n) continue;
      gl.bufferData(gl.ARRAY_BUFFER, buf.a, gl.STREAM_DRAW, 0, buf.n);
      gl.drawArrays(mode, 0, buf.n / FLOATS_PER_VERTEX);
      this.drawCalls++;
      buf.n = 0;
    }
  }

  // State setters only record (and mark the state key stale); batches are
  // split when a primitive starts with different state than the batch it
  // would join.
  stateChanged() { this.stateDirty = true; }

  // ---- immediate mode ----
  // Vertices go straight into flat Float32Array buffers: the primitive being
  // built (prim) and the batches, which grow as needed.
  begin(mode) {
    if (this.stateDirty) {
      this.stateDirty = false;
      const k = this.stateKey();
      if (k !== this.batchKey) {
        this.flush();
        this.batchKey = k;
        this.batchState = this.snapshot();
      }
    }
    this.prim = mode;
    this.primVerts.n = 0;
    const p = this.proj(), mv = this.mv();
    if (p !== this.mvpOf[0] || mv !== this.mvpOf[1]) {   // matrices are replaced, never edited
      this.mvp = mul(p, mv);
      this.mvpOf = [p, mv];
    }
    this.tm = this.texm();
  }

  vertex(x, y, z, w = 1) {
    if (this.prim === null) return;
    const m = this.mvp, t = this.tm, uv = this.uv, c = this.color;
    const b = this.primVerts, o = grow(b, FLOATS_PER_VERTEX), a = b.a;
    a[o] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
    a[o + 1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
    a[o + 2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
    a[o + 3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
    a[o + 4] = c[0]; a[o + 5] = c[1]; a[o + 6] = c[2]; a[o + 7] = c[3];
    a[o + 8] = t[0] * uv[0] + t[4] * uv[1] + t[12];
    a[o + 9] = t[1] * uv[0] + t[5] * uv[1] + t[13];
  }

  end() {
    const mode = this.prim;
    this.prim = null;
    if (mode === null) return;
    const v = this.primVerts.a, n = this.primVerts.n / FLOATS_PER_VERTEX;
    const put = (buf, i) => {
      const o = grow(buf, FLOATS_PER_VERTEX);   // first: growing replaces buf.a
      buf.a.set(v.subarray(i * FLOATS_PER_VERTEX, (i + 1) * FLOATS_PER_VERTEX), o);
    };
    const { tris, lines, points } = this.batch;
    const tri = (i, j, k) => { put(tris, i); put(tris, j); put(tris, k); };
    switch (mode) {
      case 4: for (let i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2); break;                       // TRIANGLES
      case 5: for (let i = 0; i + 2 < n; i++) i & 1 ? tri(i + 1, i, i + 2) : tri(i, i + 1, i + 2); break; // TRIANGLE_STRIP
      case 6: case 9: for (let i = 1; i + 1 < n; i++) tri(0, i, i + 1); break;                       // FAN, POLYGON
      case 7: for (let i = 0; i + 3 < n; i += 4) { tri(i, i + 1, i + 2); tri(i, i + 2, i + 3); } break; // QUADS
      case 8: for (let i = 0; i + 3 < n; i += 2) { tri(i, i + 1, i + 3); tri(i, i + 3, i + 2); } break; // QUAD_STRIP
      case 1: for (let i = 0; i + 1 < n; i += 2) { put(lines, i); put(lines, i + 1); } break;       // LINES
      case 3: case 2:                                                                                // LINE_STRIP, LINE_LOOP
        for (let i = 0; i + 1 < n; i++) { put(lines, i); put(lines, i + 1); }
        if (mode === 2 && n > 2) { put(lines, n - 1); put(lines, 0); }
        break;
      case 0: for (let i = 0; i < n; i++) put(points, i); break;                                    // POINTS
    }
  }

  // ---- 1D evaluators ----
  map1(target, u1, u2, stride, order, ptr) {
    const comps = { [E.MAP1_VERTEX_3]: 3, [E.MAP1_VERTEX_4]: 4, [E.MAP1_COLOR_4]: 4, [E.MAP1_TEXTURE_COORD_2]: 2 }[target] || 3;
    const dv = this.dv();
    const pts = [];
    for (let i = 0; i < order; i++) {
      const pt = [];
      for (let k = 0; k < comps; k++) pt.push(dv.getFloat32(ptr + 4 * (i * stride + k), true));
      pts.push(pt);
    }
    this.maps.set(target, { u1, u2, order, points: pts });
  }

  evalMap(m, u) {
    const t = (u - m.u1) / (m.u2 - m.u1), n = m.order - 1, out = new Array(m.points[0].length).fill(0);
    let binom = 1;
    for (let i = 0; i <= n; i++) {
      const b = binom * t ** i * (1 - t) ** (n - i);
      m.points[i].forEach((c, k) => { out[k] += b * c; });
      binom = binom * (n - i) / (i + 1);
    }
    return out;
  }

  evalCoord1(u) {
    const c = this.maps.get(E.MAP1_COLOR_4);
    if (c && this.enabled.has(E.MAP1_COLOR_4)) this.color = this.evalMap(c, u);
    const t = this.maps.get(E.MAP1_TEXTURE_COORD_2);
    if (t && this.enabled.has(E.MAP1_TEXTURE_COORD_2)) this.uv = this.evalMap(t, u);
    const v4 = this.maps.get(E.MAP1_VERTEX_4);
    if (v4 && this.enabled.has(E.MAP1_VERTEX_4)) { this.vertex(...this.evalMap(v4, u)); return; }
    const v3 = this.maps.get(E.MAP1_VERTEX_3);
    if (v3 && this.enabled.has(E.MAP1_VERTEX_3)) this.vertex(...this.evalMap(v3, u));
  }

  // ---- textures ----
  texImage(target, level, internal, w, h, format, type, ptr) {
    if (level !== 0 || !this.bound) return;
    this.flush();   // pending primitives may sample this texture's old contents
    const gl = this.gl, src = new Uint8Array(this.memory.buffer);
    const rgba = new Uint8Array(w * h * 4);
    const n = w * h;
    const read = { 0x1907: 3, 0x1908: 4, 0x1909: 1, 0x190a: 2, 0x1906: 1, 0x80e0: 3, 0x80e1: 4 }[format] || 4;
    for (let i = 0; i < n; i++) {
      const s = ptr + i * read, d = i * 4;
      switch (format) {
        case 0x1907: rgba[d] = src[s]; rgba[d + 1] = src[s + 1]; rgba[d + 2] = src[s + 2]; rgba[d + 3] = 255; break;
        case 0x80e0: rgba[d] = src[s + 2]; rgba[d + 1] = src[s + 1]; rgba[d + 2] = src[s]; rgba[d + 3] = 255; break;
        case 0x80e1: rgba[d] = src[s + 2]; rgba[d + 1] = src[s + 1]; rgba[d + 2] = src[s]; rgba[d + 3] = src[s + 3]; break;
        case 0x1909: rgba[d] = rgba[d + 1] = rgba[d + 2] = src[s]; rgba[d + 3] = 255; break;
        case 0x190a: rgba[d] = rgba[d + 1] = rgba[d + 2] = src[s]; rgba[d + 3] = src[s + 1]; break;
        case 0x1906: rgba[d] = rgba[d + 1] = rgba[d + 2] = 255; rgba[d + 3] = src[s]; break;
        default: rgba[d] = src[s]; rgba[d + 1] = src[s + 1]; rgba[d + 2] = src[s + 2]; rgba[d + 3] = src[s + 3];
      }
    }
    if (internal === 3 || internal === 0x1907) for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    gl.bindTexture(gl.TEXTURE_2D, this.textures.get(this.bound));
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  // ---- wglUseFontOutlines ----
  // Each character becomes a display list that draws the glyph's filled shape
  // (from a 2D-canvas atlas of the same font, alpha-tested) with the baseline
  // at y = 0, then advances the pen by the glyph's cell width. Windows scales
  // these glyphs so that 1.0 is the font's cell height (ascent + descent), not
  // its em: measured against the real wglUseFontOutlines
  // (tools/fontmetrics), every metric matches after dividing by it.
  fontOutlines(face, weight, italic, first, count, listBase, extrusion, gmf) {
    const PX = 64, CELL = 96, cols = 16;
    const rows = Math.ceil(count / cols);
    const atlas = document.createElement('canvas');
    atlas.width = cols * CELL;
    atlas.height = rows * CELL;
    const c2d = atlas.getContext('2d');
    const font = size => `${italic ? 'italic ' : ''}${weight || 400} ${size}px "${face || 'Arial'}", Arial, sans-serif`;
    c2d.font = font(1000);   // the cell height in ems, measured large so it isn't rounded
    const fm = c2d.measureText('x');
    const UNIT = PX * (fm.fontBoundingBoxAscent + fm.fontBoundingBoxDescent) / 1000;   // canvas px per output unit
    c2d.font = font(PX);
    c2d.fillStyle = '#fff';
    c2d.textBaseline = 'alphabetic';
    const dv = this.dv();
    const baseY = CELL - 24;
    for (let i = 0; i < count; i++) {
      const ch = String.fromCharCode(first + i);
      const m = c2d.measureText(ch);
      const cx = (i % cols) * CELL + 16, cy = Math.floor(i / cols) * CELL + baseY;
      c2d.fillText(ch, cx, cy);
      const left = -m.actualBoundingBoxLeft, right = m.actualBoundingBoxRight;
      const asc = m.actualBoundingBoxAscent, desc = m.actualBoundingBoxDescent;
      const glyph = {
        u0: (cx - 16) / atlas.width, u1: (cx - 16 + CELL) / atlas.width,
        v0: (cy - baseY) / atlas.height, v1: (cy - baseY + CELL) / atlas.height,
        x0: -16 / UNIT, x1: (CELL - 16) / UNIT, y0: -(CELL - baseY) / UNIT, y1: baseY / UNIT,
        advance: m.width / UNIT,
      };
      this.listGlyphs.set(listBase + i, glyph);
      if (gmf) {
        const vals = [(right - left) / UNIT, (asc + desc) / UNIT, left / UNIT, asc / UNIT, glyph.advance, 0];
        vals.forEach((v, k) => dv.setFloat32(gmf + 24 * i + 4 * k, v, true));
      }
    }
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlas);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    for (let i = 0; i < count; i++) this.listGlyphs.get(listBase + i).tex = tex;
  }

  callLists(n, type, ptr) {
    const dv = this.dv();
    const size = type === 0x1403 || type === 0x1402 ? 2 : type === 0x1404 || type === 0x1405 ? 4 : 1;
    const codes = [];
    for (let i = 0; i < n; i++) {
      const p = ptr + i * size;
      codes.push(size === 1 ? dv.getUint8(p) : size === 2 ? dv.getUint16(p, true) : dv.getUint32(p, true));
    }
    for (const code of codes) {
      const g = this.listGlyphs.get(this.listBase + code);
      if (!g) continue;
      const key = this.stateKey() + '|glyph' + this.listBase;
      if (key !== this.batchKey) {
        this.flush();
        this.batchKey = key;
        this.batchState = { ...this.snapshot(), tex: g.tex, alpha: [0x204, 0.5] };
      }
      this.stateDirty = true;    // the glyph batch's key isn't the plain state key
      this.prim = 7;             // QUADS
      this.primVerts.n = 0;
      this.mvp = mul(this.proj(), this.mv());
      this.mvpOf = [null, null];
      this.tm = ident();
      const saveUv = this.uv;
      this.uv = [g.u0, g.v1]; this.vertex(g.x0, g.y0, 0);
      this.uv = [g.u1, g.v1]; this.vertex(g.x1, g.y0, 0);
      this.uv = [g.u1, g.v0]; this.vertex(g.x1, g.y1, 0);
      this.uv = [g.u0, g.v0]; this.vertex(g.x0, g.y1, 0);
      this.uv = saveUv;
      this.end();
      const s = this.stacks[E.MODELVIEW], t = ident();
      t[12] = g.advance;
      s[s.length - 1] = mul(s[s.length - 1], t);
    }
  }

  // ---- the import object for engine.wasm ----
  imports(host) {
    const gl = this.gl;
    // Guest pointers are only byte-aligned (the engine's matrices live in
    // stack arrays at arbitrary offsets), so all access goes through DataView.
    const readF64s = (p, n) => {
      const dv = this.dv(), out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getFloat64(p + 8 * i, true);
      return out;
    };
    return {
      gl_glBegin: m => this.begin(m),
      gl_glEnd: () => this.end(),
      gl_glVertex3f: (x, y, z) => this.vertex(x, y, z),
      gl_glTexCoord2f: (s, t) => { this.uv = [s, t]; },
      gl_glColor4f: (r, g, b, a) => { this.color = [r, g, b, a]; },
      gl_glEnable: (cap, on) => { on ? this.enabled.add(cap) : this.enabled.delete(cap); this.stateChanged(); },
      gl_glBindTexture: (t, id) => { this.bound = id; this.stateChanged(); },
      gl_glAlphaFunc: (fn, ref) => { this.alpha = [fn, ref]; this.stateChanged(); },
      gl_glBlendFunc: (s, d) => { this.blend = [s, d]; this.stateChanged(); },
      gl_glFrontFace: m => { this.front = m; this.stateChanged(); },
      gl_glClearColor: (r, g, b, a) => gl.clearColor(r, g, b, a),
      gl_glClear: mask => {
        this.flush();
        this.ensureTarget();   // the window may have been resized
        gl.viewport(...this.viewport);
        gl.depthMask(true);
        gl.clear(mask & (gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT));
      },
      gl_glFlush: () => this.flush(),
      gl_glMatrixMode: m => { this.mode = m; },
      gl_glLoadIdentity: () => { this.top = ident(); },
      gl_glPushMatrix: () => { const s = this.stacks[this.mode]; s.push(new Float64Array(s[s.length - 1])); },
      gl_glPopMatrix: () => { const s = this.stacks[this.mode]; if (s.length > 1) s.pop(); },
      gl_glTranslatef: (x, y, z) => { const m = ident(); m[12] = x; m[13] = y; m[14] = z; this.top = mul(this.top, m); },
      gl_glScalef: (x, y, z) => { const m = ident(); m[0] = x; m[5] = y; m[10] = z; this.top = mul(this.top, m); },
      gl_glRotatef: (deg, x, y, z) => {
        const l = Math.hypot(x, y, z) || 1;
        x /= l; y /= l; z /= l;
        const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), t = 1 - c;
        const m = new Float64Array([
          t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0,
          t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0,
          t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0,
          0, 0, 0, 1]);
        this.top = mul(this.top, m);
      },
      gl_glMultMatrixd: p => { this.top = mul(this.top, readF64s(p, 16)); },
      gl_glOrtho: (l, r, b, t, n, f) => {
        const m = ident();
        m[0] = 2 / (r - l); m[5] = 2 / (t - b); m[10] = -2 / (f - n);
        m[12] = -(r + l) / (r - l); m[13] = -(t + b) / (t - b); m[14] = -(f + n) / (f - n);
        this.top = mul(this.top, m);
      },
      gl_glViewport: (x, y, w, h) => { this.viewport = [x, y, w, h]; this.stateChanged(); },
      gl_glGetv: (pname, p, kind) => {
        let vals;
        if (pname === E.MODELVIEW_MATRIX) vals = this.mv();
        else if (pname === E.PROJECTION_MATRIX) vals = this.proj();
        else if (pname === E.VIEWPORT) vals = this.viewport;
        else { host.log(`[gl] glGet 0x${pname.toString(16)} not emulated`); vals = [0, 0, 0, 0]; }
        const dv = this.dv();
        for (let i = 0; i < vals.length; i++) {
          if (kind === 2) dv.setFloat64(p + 8 * i, vals[i], true);
          else if (kind === 1) dv.setFloat32(p + 4 * i, vals[i], true);
          else dv.setInt32(p + 4 * i, vals[i], true);
        }
      },
      gl_glMap1f: (t, u1, u2, stride, order, p) => this.map1(t, u1, u2, stride, order, p),
      gl_glEvalCoord1f: u => this.evalCoord1(u),
      gl_glRasterPos2i: () => {},
      gl_glLineWidth: () => {},
      gl_glPushAttrib: () => this.attribStack.push({
        enabled: new Set(this.enabled), color: [...this.color], blend: [...this.blend], alpha: [...this.alpha],
        front: this.front, viewport: [...this.viewport],
      }),
      gl_glPopAttrib: () => {
        const s = this.attribStack.pop();
        if (s) Object.assign(this, s);
        this.stateChanged();
      },
      gl_glGenTextures: (n, p) => {
        const dv = this.dv();
        for (let i = 0; i < n; i++) {
          const id = this.nextTexture++;
          const t = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          this.textures.set(id, t);
          dv.setUint32(p + 4 * i, id, true);
        }
      },
      gl_glTexImage2D: (target, level, internal, w, h, border, format, type, p) =>
        this.texImage(target, level, internal, w, h, format, type, p),
      gl_glBuildMipmaps: (target, comps, w, h, format, type, p) => this.texImage(target, 0, comps, w, h, format, type, p),
      gl_glTexParameterf: (target, pname, v) => {
        const t = this.textures.get(this.bound);
        if (!t) return;
        gl.bindTexture(gl.TEXTURE_2D, t);
        const map = { 0x2600: gl.NEAREST, 0x2601: gl.LINEAR, 0x2700: gl.NEAREST_MIPMAP_NEAREST, 0x2701: gl.LINEAR_MIPMAP_NEAREST,
          0x2702: gl.NEAREST_MIPMAP_LINEAR, 0x2703: gl.LINEAR_MIPMAP_LINEAR, 0x2900: gl.CLAMP_TO_EDGE, 0x812f: gl.CLAMP_TO_EDGE, 0x2901: gl.REPEAT };
        if ([0x2800, 0x2801, 0x2802, 0x2803].includes(pname) && map[v] !== undefined) gl.texParameteri(gl.TEXTURE_2D, pname, map[v]);
      },
      gl_gluProject: (x, y, z, model, proj, view, wx, wy, wz) => {
        const dv = this.dv();
        const m = readF64s(model, 16), p = readF64s(proj, 16);
        const v = [0, 1, 2, 3].map(i => dv.getInt32(view + 4 * i, true));
        const e = [0, 1, 2, 3].map(r => m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]);
        const c = [0, 1, 2, 3].map(r => p[r] * e[0] + p[4 + r] * e[1] + p[8 + r] * e[2] + p[12 + r] * e[3]);
        if (!c[3]) return 0;
        const nx = c[0] / c[3], ny = c[1] / c[3], nz = c[2] / c[3];
        dv.setFloat64(wx, v[0] + (1 + nx) * v[2] / 2, true);
        dv.setFloat64(wy, v[1] + (1 + ny) * v[3] / 2, true);
        dv.setFloat64(wz, (1 + nz) / 2, true);
        return 1;
      },
      gl_glGenLists: n => { const b = this.lists; this.lists += n; return b; },
      gl_glCallLists: (n, type, p) => this.callLists(n, type, p),
      gl_glListBase: b => { this.listBase = b; },
      gl_glFontOutlines: (face, weight, italic, first, count, base, extrusion, gmf) =>
        this.fontOutlines(face ? host.cstr(face) : 'Arial', weight, italic, first, count, base, extrusion, gmf),
    };
  }
}
